from __future__ import annotations

import subprocess
import sys
import unittest
import importlib.util
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from cxr_thesis.objective4 import GradCAM, integrated_gradients


_LOCK_SPEC = importlib.util.spec_from_file_location(
    "lock_objective4_xai_protocol",
    Path(__file__).parents[1] / "scripts" / "lock_objective4_xai_protocol.py",
)
assert _LOCK_SPEC is not None and _LOCK_SPEC.loader is not None
_LOCK_MODULE = importlib.util.module_from_spec(_LOCK_SPEC)
_LOCK_SPEC.loader.exec_module(_LOCK_MODULE)

_PUBLISH_SPEC = importlib.util.spec_from_file_location(
    "publish_objective4_xai_protocol",
    Path(__file__).parents[1] / "scripts" / "publish_objective4_xai_protocol.py",
)
assert _PUBLISH_SPEC is not None and _PUBLISH_SPEC.loader is not None
_PUBLISH_MODULE = importlib.util.module_from_spec(_PUBLISH_SPEC)
_PUBLISH_SPEC.loader.exec_module(_PUBLISH_MODULE)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4 + 9, 2)

    def forward(self, image, clinical):
        encoded = self.pool(torch.relu(self.features(image))).flatten(1)
        return self.classifier(torch.cat([encoded, clinical], dim=1))


class Objective4Tests(unittest.TestCase):
    def test_gradcam_and_integrated_gradients(self) -> None:
        torch.manual_seed(42)
        model = TinyModel().eval()
        image = torch.randn(2, 3, 16, 16)
        clinical = torch.randn(2, 9)
        with GradCAM(model, model.features) as explainer:
            cam, _ = explainer(image, clinical, 0)
        integrated, _ = integrated_gradients(
            model, image, clinical, 0, steps=4
        )
        self.assertEqual(tuple(cam.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(integrated.shape), tuple(cam.shape))
        self.assertTrue(torch.isfinite(cam).all())
        self.assertTrue(torch.isfinite(integrated).all())

    def test_smoke_cli(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "smoke_objective4_xai.py"),
                "--batch-size",
                "1",
                "--ig-steps",
                "4",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OBJECTIVE 4 XAI METHOD SMOKE SUCCESSFUL", result.stdout)

    def test_validation_only_balanced_protocol_selection(self) -> None:
        rows = []
        for label_index, label in enumerate(_LOCK_MODULE.LABELS):
            for case_index in range(4):
                row = {
                    "patient_id": f"p-{label_index}-{case_index}",
                    "image_id": f"i-{label_index}-{case_index}",
                    "image_path": f"images/{label_index}-{case_index}.png",
                    "split": "val",
                }
                row.update(
                    {
                        f"label_{name}": int(name == label)
                        for name in _LOCK_MODULE.LABELS
                    }
                )
                rows.append(row)
        cohort = _LOCK_MODULE.select_cohort(
            pd.DataFrame(rows), seed=42, cases_per_label=2
        )
        self.assertEqual(len(cohort), 24)
        self.assertEqual(cohort["patient_id"].nunique(), 24)
        self.assertEqual(cohort["image_id"].nunique(), 24)
        self.assertEqual(set(cohort["split"]), {"val"})
        self.assertEqual(set(cohort["xai_target_label"].value_counts()), {2})

        invalid = pd.DataFrame(rows)
        invalid.loc[0, "split"] = "test"
        with self.assertRaises(ValueError):
            _LOCK_MODULE.select_cohort(invalid, seed=42, cases_per_label=2)

    def test_public_protocol_validation_rejects_test_access(self) -> None:
        protocol = {
            "artifact": "Objective 4 quantitative XAI protocol lock",
            "status": "locked_before_explanation_generation",
            "objective": 4,
            "model": "densenet121",
            "expected_checkpoint_sha256": "checkpoint",
            "private_xai_cohort_sha256": "cohort",
            "cohort": {
                "split": "val",
                "cases": 240,
                "unique_patients": 240,
                "unique_images": 240,
                "cases_per_target_label": 20,
                "target_label_counts": {f"label-{index}": 20 for index in range(12)},
                "predictions_used_for_selection": False,
                "risk_scores_used_for_selection": False,
            },
            "protections": {
                "test_manifest_opened": False,
                "test_labels_accessed": False,
                "test_evaluated": False,
                "manual_masking_required": False,
                "private_manifest_allowed_for_public_upload": False,
                "medical_images_public": False,
                "case_level_explanations_public": False,
            },
        }
        checks = _PUBLISH_MODULE.validate_protocol(
            protocol,
            expected_private_hash="cohort",
            expected_checkpoint_hash="checkpoint",
        )
        self.assertTrue(all(checks.values()))
        protocol["protections"]["test_manifest_opened"] = True
        with self.assertRaises(RuntimeError):
            _PUBLISH_MODULE.validate_protocol(
                protocol,
                expected_private_hash="cohort",
                expected_checkpoint_hash="checkpoint",
            )


if __name__ == "__main__":
    unittest.main()
