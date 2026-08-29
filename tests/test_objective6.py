from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

import torch

from cxr_thesis.objective6.models import DenseNetTransformerReportGenerator
from cxr_thesis.objective6.text import ReportVocabulary, normalise_report, tokenise_report


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


if __name__ == "__main__":
    unittest.main()
