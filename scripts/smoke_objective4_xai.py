#!/usr/bin/env python3
"""Smoke-test Objective 4 Grad-CAM and integrated gradients."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_thesis.objective4 import GradCAM, integrated_gradients


class TinyImageClinicalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8 + 9, 12)

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        image_features = self.pool(self.features(image)).flatten(1)
        return self.classifier(torch.cat([image_features, clinical], dim=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--ig-steps", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(42)
    model = TinyImageClinicalModel().eval()
    image = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    clinical = torch.randn(args.batch_size, 9)
    with GradCAM(model, model.features[-2]) as explainer:
        grad_cam, logits = explainer(image, clinical, 0)
    integrated, ig_logits = integrated_gradients(
        model, image, clinical, 0, steps=args.ig_steps
    )
    checks = {
        "grad_cam_shape": tuple(grad_cam.shape)
        == (args.batch_size, 1, args.image_size, args.image_size),
        "integrated_gradients_shape": tuple(integrated.shape)
        == tuple(grad_cam.shape),
        "logits_shape": tuple(logits.shape) == (args.batch_size, 12),
        "consistent_logits": torch.allclose(logits, ig_logits, atol=1e-5),
        "grad_cam_finite": torch.isfinite(grad_cam).all().item(),
        "integrated_gradients_finite": torch.isfinite(integrated).all().item(),
        "grad_cam_bounded": bool((grad_cam >= 0).all() and (grad_cam <= 1).all()),
        "integrated_gradients_bounded": bool(
            (integrated >= 0).all() and (integrated <= 1).all()
        ),
    }
    print("--- OBJECTIVE 4 XAI SMOKE ---")
    for name, passed in checks.items():
        print(name + ":", passed)
    if not all(checks.values()):
        raise RuntimeError(f"Objective 4 smoke failed: {checks}")
    print("Medical images accessed: False")
    print("Test manifest opened: False")
    print("Test labels accessed: False")
    print("OBJECTIVE 4 XAI METHOD SMOKE SUCCESSFUL")


if __name__ == "__main__":
    main()
