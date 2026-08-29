from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cxr_thesis.objective2.data import ImageClassificationDataset
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective5.adaptation import (
    initialise_shared_label_densenet,
    set_adaptation_phase,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_objective5_locked_tests import patient_bootstrap
from lock_objective5_selected_candidates import fit_temperature


class Objective5AdaptationTests(unittest.TestCase):
    def test_shared_output_rows_are_copied_exactly(self) -> None:
        source_labels = ["A", "B", "C"]
        source = build_classifier("densenet121", 3, pretrained=False)
        with torch.no_grad():
            source.classifier[-1].weight.copy_(
                torch.arange(source.classifier[-1].weight.numel()).reshape_as(
                    source.classifier[-1].weight
                )
            )
            source.classifier[-1].bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        checkpoint = {
            "label_names": source_labels,
            "model_state": source.state_dict(),
            "model_config": {"clinical_dim": 9, "dropout": 0.2},
        }
        target, indices = initialise_shared_label_densenet(checkpoint, ["C", "A"])
        self.assertEqual(indices, [2, 0])
        self.assertTrue(
            torch.equal(
                target.classifier[-1].weight,
                source.classifier[-1].weight[[2, 0]],
            )
        )
        self.assertTrue(
            torch.equal(target.classifier[-1].bias, torch.tensor([3.0, 1.0]))
        )

    def test_phase_policy_only_unfreezes_locked_parameters(self) -> None:
        model = build_classifier("densenet121", 6, pretrained=False)
        set_adaptation_phase(model, "head_warmup")
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.encoder.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.classifier.parameters())
        )
        set_adaptation_phase(model, "final_block")
        self.assertTrue(
            any(
                parameter.requires_grad
                for parameter in model.encoder.features.denseblock4.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.encoder.features.denseblock3.parameters()
            )
        )

    def test_locked_augmentation_does_not_flip(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "image_path": "unused",
                    "label_A": 1,
                    "age": 50,
                    "sex": "F",
                    "view": "PA",
                }
            ]
        )
        dataset = ImageClassificationDataset(
            frame,
            ["label_A"],
            augment=True,
            augmentation_profile="objective5_locked",
            horizontal_flip_probability=0.0,
        )
        image = np.tile(np.arange(32, dtype=np.float32), (32, 1)) / 31.0
        augmented = dataset._augment(image, 0)
        self.assertGreater(
            float(augmented[:, -1].mean()), float(augmented[:, 0].mean())
        )

    def test_training_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "train_objective5_adaptation.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--source-checkpoint", result.stdout)

    def test_temperature_scaling_reduces_nll(self) -> None:
        logits = np.asarray([[8.0], [-8.0], [4.0], [-4.0]], dtype=np.float64)
        targets = np.asarray([[1.0], [0.0], [0.0], [1.0]], dtype=np.float64)
        temperature, before, after = fit_temperature(logits, targets)
        self.assertGreater(temperature, 1.0)
        self.assertLessEqual(after, before)

    def test_selection_lock_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "lock_objective5_selected_candidates.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--chexpert-checkpoint", result.stdout)

    def test_selection_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "publish_objective5_selection_lock.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-summary-sha256", result.stdout)

    def test_locked_test_evaluation_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "evaluate_objective5_locked_tests.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--chexpert-test", result.stdout)
        self.assertIn("--private-hf-path", result.stdout)

    def test_objective5_bootstrap_is_patient_clustered(self) -> None:
        rng = np.random.default_rng(42)
        probabilities = rng.uniform(0.05, 0.95, size=(24, 6))
        targets = np.asarray(
            [[(row + label) % 3 == 0 for label in range(6)] for row in range(24)],
            dtype=np.int8,
        )
        result = patient_bootstrap(
            probabilities,
            targets,
            np.full(6, 0.5),
            np.asarray([f"P{row:03d}" for row in range(24)]),
            replicates=10,
            seed=42,
        )
        self.assertEqual(result["method"], "patient-cluster percentile bootstrap")
        self.assertEqual(result["replicates"], 10)
        self.assertEqual(set(result["macro_95_ci"]), {"auroc", "auprc", "f1", "brier", "ece"})

    def test_locked_test_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(
                    repository
                    / "scripts"
                    / "publish_objective5_locked_test_results.py"
                ),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-lock-sha256", result.stdout)


if __name__ == "__main__":
    unittest.main()
