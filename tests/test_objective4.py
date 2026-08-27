from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

from cxr_thesis.objective4 import GradCAM, integrated_gradients


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


if __name__ == "__main__":
    unittest.main()
