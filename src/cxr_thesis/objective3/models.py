"""Matched classical and variational-quantum graph heads for Objective 3."""

from __future__ import annotations

import math

import torch
from torch import nn


def bottleneck_parameter_count(module: nn.Module) -> int:
    """Count trainable parameters belonging to one bottleneck."""

    return sum(parameter.numel() for parameter in module.parameters())


class ClassicalMatchedBottleneck(nn.Module):
    """A 24-parameter classical control for the four-qubit circuit."""

    def __init__(self, features: int = 4) -> None:
        super().__init__()
        if features != 4:
            raise ValueError("The frozen matched control requires four features")
        self.mix = nn.Linear(features, features)
        self.gate = nn.Parameter(torch.ones(features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 4:
            raise ValueError("Classical bottleneck expects [batch, 4]")
        return torch.tanh(self.mix(inputs)) * torch.sigmoid(self.gate)


class QuantumBottleneck(nn.Module):
    """Four-qubit, two-layer variational circuit with 24 trainable angles."""

    def __init__(self, qubits: int = 4, layers: int = 2) -> None:
        super().__init__()
        if qubits != 4 or layers != 2:
            raise ValueError("The frozen Objective 3 circuit is 4 qubits x 2 layers")
        try:
            import pennylane as qml
        except ImportError as error:
            raise RuntimeError(
                "PennyLane 0.45.1 is required for the quantum bottleneck"
            ) from error

        wires = tuple(range(qubits))
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=wires)
            return [qml.expval(qml.PauliZ(wire)) for wire in wires]

        weight_shapes = {"weights": (layers, qubits, 3)}
        self.layer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes,
            init_method=torch.nn.init.normal_,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 4:
            raise ValueError("Quantum bottleneck expects [batch, 4]")
        output = self.layer(inputs)
        return output.to(dtype=inputs.dtype)


class ClassicalReuploadingBottleneck(nn.Module):
    """A 36-parameter classical control for the re-uploading circuit."""

    def __init__(self, features: int = 4) -> None:
        super().__init__()
        if features != 4:
            raise ValueError("The enhanced matched control requires four features")
        # 4x4 weights + 4 biases + 4x4 bias-free weights = 36 parameters.
        self.input_mix = nn.Linear(features, features)
        self.output_mix = nn.Linear(features, features, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 4:
            raise ValueError("Classical re-uploading bottleneck expects [batch, 4]")
        return torch.tanh(self.output_mix(torch.tanh(self.input_mix(inputs))))


class QuantumReuploadingBottleneck(nn.Module):
    """Four-qubit circuit with three data re-uploading blocks and 36 angles."""

    def __init__(self, qubits: int = 4, layers: int = 3) -> None:
        super().__init__()
        if qubits != 4 or layers != 3:
            raise ValueError(
                "The enhanced Objective 3 circuit is 4 qubits x 3 re-uploading blocks"
            )
        try:
            import pennylane as qml
        except ImportError as error:
            raise RuntimeError(
                "PennyLane 0.45.1 is required for the quantum bottleneck"
            ) from error

        wires = tuple(range(qubits))
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for block in range(layers):
                qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
                qml.StronglyEntanglingLayers(
                    weights[block : block + 1], wires=wires
                )
            return [qml.expval(qml.PauliZ(wire)) for wire in wires]

        weight_shapes = {"weights": (layers, qubits, 3)}
        self.layer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes,
            init_method=torch.nn.init.normal_,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 4:
            raise ValueError("Quantum re-uploading bottleneck expects [batch, 4]")
        output = self.layer(inputs)
        return output.to(dtype=inputs.dtype)


class HybridGraphHead(nn.Module):
    """Residual classifier using either quantum or matched classical features."""

    def __init__(
        self,
        labels: int,
        *,
        embedding_dim: int = 160,
        bottleneck: str = "quantum",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if labels <= 0 or embedding_dim <= 0:
            raise ValueError("labels and embedding_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        normalised = bottleneck.strip().lower().replace("-", "_")
        self.embedding_dim = int(embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, 4)
        if normalised == "quantum":
            self.bottleneck = QuantumBottleneck()
        elif normalised in {"classical", "classical_matched"}:
            self.bottleneck = ClassicalMatchedBottleneck()
        else:
            raise ValueError("bottleneck must be quantum or classical_matched")
        self.bottleneck_name = normalised
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim + 4, labels),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Hybrid graph head expects [batch, {self.embedding_dim}]")
        angles = math.pi * torch.tanh(self.input_projection(embeddings))
        enhanced = self.bottleneck(angles)
        return self.classifier(torch.cat([embeddings, enhanced], dim=1))


class EnhancedHybridGraphHead(nn.Module):
    """v1.1 paired head using re-uploading and gated residual fusion."""

    def __init__(
        self,
        labels: int,
        *,
        embedding_dim: int = 160,
        bottleneck: str = "quantum",
        dropout: float = 0.2,
        initial_fusion_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if labels <= 0 or embedding_dim <= 0:
            raise ValueError("labels and embedding_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < initial_fusion_scale < 1.0:
            raise ValueError("initial_fusion_scale must be in (0, 1)")

        normalised = bottleneck.strip().lower().replace("-", "_")
        self.embedding_dim = int(embedding_dim)
        self.input_projection = nn.Linear(embedding_dim, 4)
        if normalised == "quantum":
            self.bottleneck = QuantumReuploadingBottleneck()
        elif normalised in {"classical", "classical_matched"}:
            self.bottleneck = ClassicalReuploadingBottleneck()
        else:
            raise ValueError("bottleneck must be quantum or classical_matched")
        self.bottleneck_name = normalised
        self.output_projection = nn.Linear(4, embedding_dim, bias=False)
        fusion_logit = math.log(initial_fusion_scale / (1.0 - initial_fusion_scale))
        self.fusion_logit = nn.Parameter(torch.tensor(fusion_logit))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, labels),
        )

    @property
    def fusion_scale(self) -> torch.Tensor:
        """Return the bounded learned residual scale."""

        return torch.sigmoid(self.fusion_logit)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Enhanced hybrid graph head expects [batch, {self.embedding_dim}]"
            )
        angles = math.pi * torch.tanh(self.input_projection(embeddings))
        bottleneck_features = self.bottleneck(angles)
        residual = self.output_projection(bottleneck_features)
        enhanced = embeddings + self.fusion_scale * residual
        return self.classifier(enhanced)
