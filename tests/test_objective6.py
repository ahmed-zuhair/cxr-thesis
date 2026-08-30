from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path
import hashlib
import os
import tempfile

import torch
import numpy as np
from PIL import Image

from cxr_thesis.objective6.models import DenseNetTransformerReportGenerator
from cxr_thesis.objective6.data import collate_reports
from cxr_thesis.objective6.cohorts import (
    derive_padchest_age,
    patient_partition,
)
from cxr_thesis.objective6.text import ReportVocabulary, normalise_report, tokenise_report
import pandas as pd
from cxr_thesis.objective2.models import build_classifier
from scripts.train_objective6_with_private_recovery import snapshot


class Objective6TextTests(unittest.TestCase):
    def test_spanish_normalisation_and_tokenisation(self) -> None:
        self.assertEqual(normalise_report("  Sin   hallazgos ÁGUDOS. "), "sin hallazgos águdos.")
        self.assertEqual(
            tokenise_report("Sin hallazgos águdos."),
            ["sin", "hallazgos", "águdos", "."],
        )

    def test_vocabulary_is_deterministic_and_round_trips(self) -> None:
        reports = ["sin hallazgos .", "sin derrame pleural .", "derrame pleural ."]
        first = ReportVocabulary.build(reports, minimum_frequency=1, maximum_size=20)
        second = ReportVocabulary.build(reversed(reports), minimum_frequency=1, maximum_size=20)
        self.assertEqual(first.tokens, second.tokens)
        encoded = first.encode("sin derrame pleural .", maximum_length=10)
        self.assertEqual(first.decode(encoded), "sin derrame pleural.")
        self.assertEqual(ReportVocabulary.from_dict(first.to_dict()), first)


class Objective6ModelTests(unittest.TestCase):
    def test_forward_and_generation_shapes_without_weight_download(self) -> None:
        model = DenseNetTransformerReportGenerator(
            32,
            d_model=32,
            heads=4,
            layers=1,
            feedforward_dim=64,
            maximum_length=12,
            pretrained=False,
        )
        model.eval()
        image = torch.rand(2, 3, 64, 64)
        clinical = torch.rand(2, 9)
        input_ids = torch.tensor([[1, 5, 6, 2], [1, 7, 8, 2]])
        with torch.no_grad():
            output = model(image, clinical, input_ids)
            generated = model.generate(image, clinical, maximum_length=5)
        self.assertEqual(output["report_logits"].shape, (2, 4, 32))
        self.assertEqual(output["clinical_logits"].shape, (2, 6))
        self.assertEqual(generated.shape[0], 2)
        self.assertLessEqual(generated.shape[1], 5)
        self.assertFalse(any(parameter.requires_grad for parameter in model.image_encoder.parameters()))


class Objective6CliTests(unittest.TestCase):
    def test_private_recovery_snapshot_accepts_existing_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            checkpoint = output / "last.pt"
            torch.save({"epoch_completed": 3, "test_evaluated": False}, checkpoint)
            checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (output / "last.pt.sha256").write_text(
                f"{checksum}  last.pt\n", encoding="utf-8"
            )
            target = root / "already_created"
            target.mkdir()

            paths, epoch = snapshot(output, target)

            self.assertEqual(epoch, 3)
            self.assertEqual(
                {path.name for path in paths}, {"last.pt", "last.pt.sha256"}
            )
            self.assertTrue((target / "last.pt").is_file())

    def test_read_only_audit_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "audit_objective6_report_data.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--objective5-private-root", result.stdout)

    def test_protocol_lock_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repository / "scripts" / "lock_objective6_report_protocol.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--audit-summary", result.stdout)

    def test_protocol_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repository / "scripts" / "publish_objective6_report_protocol.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--private-hf-repo", result.stdout)

    def test_training_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repository / "scripts" / "train_objective6_report_generator.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--source-checkpoint", result.stdout)

    def test_private_recovery_training_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repository / "scripts" / "train_objective6_with_private_recovery.py"), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-train-sha256", result.stdout)

    def test_validation_evaluation_lock_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "lock_objective6_validation_evaluation.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--multimodal-output", result.stdout)

    def test_validation_evaluation_publication_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "publish_objective6_validation_protocol.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--expected-protocol-sha256", result.stdout)

    def test_training_cli_writes_resumable_test_blind_checkpoint(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(4):
                image_path = root / f"image_{index}.png"
                Image.fromarray(
                    np.full((64, 64), 32 + index * 20, dtype=np.uint8)
                ).save(image_path)
                rows.append({
                    "image_path": str(image_path),
                    "patient_id": f"P{index}",
                    "study_id": f"S{index}",
                    "report": "sin hallazgos ." if index % 2 == 0 else "derrame pleural .",
                    "age": 40 + index,
                    "sex": "F" if index % 2 == 0 else "M",
                    "view": "PA",
                    "split": "train" if index < 2 else "val",
                })
            train_path = root / "train.csv"
            val_path = root / "val.csv"
            pd.DataFrame(rows[:2]).to_csv(train_path, index=False)
            pd.DataFrame(rows[2:]).to_csv(val_path, index=False)
            source = build_classifier("densenet121", 6, pretrained=False)
            checkpoint_path = root / "source.pt"
            torch.save({"model_state": source.state_dict()}, checkpoint_path)
            source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            output = root / "output"
            base = [
                sys.executable,
                str(repository / "scripts" / "train_objective6_report_generator.py"),
                "--variant", "multimodal",
                "--train-manifest", str(train_path),
                "--val-manifest", str(val_path),
                "--source-checkpoint", str(checkpoint_path),
                "--expected-source-sha256", source_hash,
                "--output-dir", str(output),
                "--batch-size", "2",
                "--workers", "0",
                "--image-size", "64",
                "--maximum-length", "12",
                "--maximum-vocabulary-size", "64",
                "--minimum-token-frequency", "1",
                "--patience", "4",
                "--no-amp",
            ]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(repository / "src")
            first = subprocess.run(
                [*base, "--epochs", "1"], cwd=repository, env=environment,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)
            recovery = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(recovery["epoch_completed"], 1)
            self.assertFalse(recovery["test_evaluated"])
            resumed = subprocess.run(
                [*base, "--epochs", "2", "--resume"], cwd=repository, env=environment,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(resumed.returncode, 0, msg=resumed.stdout + resumed.stderr)
            final = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(final["epoch_completed"], 2)
            self.assertEqual(final["resume_count"], 1)
            self.assertFalse(final["test_evaluated"])


class Objective6CohortTests(unittest.TestCase):
    def test_patient_partition_is_deterministic_and_patient_level(self) -> None:
        self.assertEqual(patient_partition("PadChest-001"), patient_partition("001"))
        self.assertIn(patient_partition("90210"), {"train", "val", "test"})

    def test_age_is_derived_from_study_year_and_birth_year(self) -> None:
        age = derive_padchest_age(
            pd.Series([20140915, 20150101, 20190101]),
            pd.Series([1930, 2010, 1800]),
        )
        self.assertEqual(float(age.iloc[0]), 84.0)
        self.assertEqual(float(age.iloc[1]), 5.0)
        self.assertTrue(pd.isna(age.iloc[2]))

    def test_report_collation_pads_only_to_batch_maximum(self) -> None:
        sample = {
            "image": torch.zeros(3, 8, 8),
            "clinical": torch.zeros(9),
            "report_ids": torch.tensor([1, 4, 2]),
        }
        longer = dict(sample)
        longer["report_ids"] = torch.tensor([1, 5, 6, 2])
        batch = collate_reports([sample, longer])
        self.assertEqual(batch["report_ids"].shape, (2, 4))
        self.assertEqual(int(batch["report_ids"][0, -1]), 0)


if __name__ == "__main__":
    unittest.main()
