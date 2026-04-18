"""
models.py
=========
Model definitions for the thesis. Starting with the CheXNet baseline.

Why DenseNet-121?
- Original CheXNet (Rajpurkar et al., 2017) used it on this exact dataset.
- Achieves strong AUC with relatively few parameters (~7M).
- Standard benchmark — easy to compare against published results.

Why ImageNet pretrained?
- Random init needs 50+ epochs and millions of CXRs to converge.
- ImageNet features transfer well even though CXRs look nothing like cats/dogs.
- Standard practice across all published medical imaging papers.
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121MultiLabel(nn.Module):
    """
    DenseNet-121 with the final classifier replaced for 14-class multi-label.

    Output: raw logits of shape (batch, 14). Apply sigmoid for probabilities.
    Loss: pairs with BCEWithLogitsLoss(pos_weight=...) — sigmoid is fused in.
    """
    def __init__(self, num_classes: int = 14, pretrained: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # DenseNet-121 ends with: features (Conv blocks) -> classifier (Linear 1024 -> 1000)
        # We replace the classifier with our multi-label head.
        in_features = backbone.classifier.in_features  # 1024
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters — useful for thesis tables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DenseNet121MultiLabel(num_classes=14)
    print(f"DenseNet121MultiLabel — {count_parameters(model):,} trainable parameters")
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}  (logits — apply sigmoid for probabilities)")
