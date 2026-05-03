"""
quantum_cnn_model.py
====================
Hybrid quantum-classical multi-label CXR classifier.

Architecture:
    Image (B, 3, 224, 224)
        |
        v
    DenseNet-121 backbone (FROZEN, ImageNet + your trained weights)
        |
        v
    Global average pooling -> (B, 1024)
        |
        v
    Linear projection 1024 -> 4 (learnable bottleneck)
        |
        v
    Tanh activation (bounds inputs to quantum circuit angle range)
        |
        v
    Quantum circuit: 4 qubits, 2 layers strongly entangling -> (B, 4)
        |
        v
    Linear classifier 4 -> 14 (with intermediate hidden layer)

The quantum circuit replaces what would otherwise be a small feedforward
network at the bottleneck. The fair comparison is against an identical
architecture where the quantum layer is replaced by a classical Linear+ReLU
operation of the same input/output dimensions, which we provide as the
ClassicalBottleneck class for ablation purposes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DenseNet121MultiLabel
from quantum_layer import QuantumLayer, N_QUBITS


class DenseNetFeatureExtractor(nn.Module):
    """
    Frozen DenseNet-121 feature extractor.
    Returns global-average-pooled 1024-dim features per image.
    """
    def __init__(self, pretrained_ckpt: str = None):
        super().__init__()
        full_model = DenseNet121MultiLabel(num_classes=14, pretrained=False)
        if pretrained_ckpt is not None:
            ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            full_model.load_state_dict(ckpt["model_state"])
            print(f"Backbone weights loaded from {pretrained_ckpt}")
            print(f"Backbone original val AUC: {ckpt.get('val_mean_auc', 'unknown')}")

        # Use the convolutional features only, not the original classifier head.
        self.features = full_model.backbone.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)
            feat = F.relu(feat, inplace=True)
            # Global average pool the (B, 1024, 7, 7) feature map to (B, 1024)
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return feat

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self


class QuantumCNN(nn.Module):
    """
    Hybrid model: frozen DenseNet -> linear projection -> quantum circuit ->
    linear classifier.
    """
    def __init__(self, pretrained_ckpt: str, num_classes: int = 14,
                 n_qubits: int = N_QUBITS, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = DenseNetFeatureExtractor(pretrained_ckpt=pretrained_ckpt)

        # Project 1024 features down to n_qubits inputs for the quantum circuit.
        # Tanh keeps inputs within [-1, 1], which maps cleanly to rotation angles
        # in the quantum circuit. We multiply by pi to span the full angle range.
        self.pre_quantum = nn.Sequential(
            nn.Linear(1024, n_qubits),
            nn.Tanh(),
        )

        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=2)

        # Post-quantum classifier. Takes n_qubits outputs to num_classes logits
        # via a hidden layer that gives the model a bit more expressive power
        # after the quantum bottleneck.
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)              # (B, 1024)
        compressed = self.pre_quantum(features)  # (B, n_qubits) in [-1, 1]
        # Scale to [-pi, pi] for full angle range of rotation gates
        scaled = compressed * 3.14159
        quantum_out = self.quantum(scaled)        # (B, n_qubits) in [-1, 1]
        # Cast quantum output to the dtype expected by the classifier
        quantum_out = quantum_out.to(features.dtype)
        return self.classifier(quantum_out)


class ClassicalCNN(nn.Module):
    """
    Ablation control: identical architecture but with a classical MLP in
    place of the quantum circuit. Used to verify any quantum advantage is
    real rather than an artifact of architectural choices.
    """
    def __init__(self, pretrained_ckpt: str, num_classes: int = 14,
                 bottleneck_dim: int = 4, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = DenseNetFeatureExtractor(pretrained_ckpt=pretrained_ckpt)
        self.pre_bottleneck = nn.Sequential(
            nn.Linear(1024, bottleneck_dim),
            nn.Tanh(),
        )
        # Classical replacement for the quantum layer: same input/output dims
        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        compressed = self.pre_bottleneck(features)
        bottlenecked = self.bottleneck(compressed)
        return self.classifier(bottlenecked)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    quantum_model = QuantumCNN(pretrained_ckpt=None, num_classes=14)
    classical_model = ClassicalCNN(pretrained_ckpt=None, num_classes=14)

    print(f"Quantum CNN trainable params: {count_trainable(quantum_model):,}")
    print(f"Classical CNN trainable params: {count_trainable(classical_model):,}")

    x = torch.randn(2, 3, 224, 224)
    yq = quantum_model(x)
    yc = classical_model(x)
    print(f"Quantum CNN output shape: {tuple(yq.shape)}")
    print(f"Classical CNN output shape: {tuple(yc.shape)}")
    print("Both models work end-to-end")
