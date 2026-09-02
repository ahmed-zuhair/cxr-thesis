"""Part 6: a quantum circuit whose entangling pattern is the graph itself.

This is what the thesis title claims and what v1.0 and v1.1 did not do. There the
quantum layer received a 160-dimensional vector produced by a frozen classical
GAT, so the graph had already been collapsed before the circuit saw anything.

Here the graph reaches the circuit. Each patch graph is coarsened to ``k``
supernodes by spatial quadrant, one qubit per supernode, and a two-qubit gate is
applied on a pair only to the extent that an edge exists between those
quadrants. Absent edges contribute nothing.

The mechanism is a weighted entangler rather than dynamic gate placement:

    CRZ(theta_ij * a_ij) on pair (i, j)

With ``a_ij = 0`` this is the identity, so the circuit structure stays fixed and
batchable while its *effect* is data-dependent. The three entangling variants are
then literally the same circuit with different adjacency inputs, which makes them
exactly parameter-matched rather than approximately so:

* ``graph``    a_ij from the real coarsened adjacency
* ``complete`` a_ij = 1 everywhere
* ``none``     a_ij = 0 everywhere

``graph`` versus ``complete`` is the ablation that matters. It asks whether the
topology carries information the circuit can use, holding every parameter and
every gate fixed. A win for ``complete`` would mean the entanglement helped but
the graph did not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

ENTANGLE_MODES = ("graph", "complete", "none")
DEFAULT_SUPERNODES = 4


# --------------------------------------------------------------------------
# coarsening
# --------------------------------------------------------------------------


def quadrant_assignment(positions: np.ndarray, supernodes: int) -> np.ndarray:
    """Assign each node to a supernode by spatial quadrant.

    Four supernodes are the chest quadrants, which keeps the qubits
    interpretable: a gate between qubits 0 and 1 means the upper-left and
    upper-right regions of that study are connected in its patch graph. The
    split is at the median of each axis so the partition adapts to the graph
    rather than to an assumed image frame.
    """

    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("positions must be [nodes, >=2]")
    if supernodes not in (2, 4):
        raise ValueError("supernodes must be 2 or 4 for quadrant coarsening")
    vertical = array[:, 0] >= np.median(array[:, 0])
    if supernodes == 2:
        return vertical.astype(np.int64)
    horizontal = array[:, 1] >= np.median(array[:, 1])
    return (vertical.astype(np.int64) * 2 + horizontal.astype(np.int64))


def coarsen(
    features: np.ndarray,
    edge_index: np.ndarray,
    positions: np.ndarray,
    supernodes: int = DEFAULT_SUPERNODES,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool node features per supernode and derive the supernode adjacency.

    Returns ``(pooled [k, F], adjacency [k, k])``. The adjacency is symmetric,
    has a zero diagonal, and is normalised by its maximum so the entangling
    angles stay in a comparable range across graphs of different densities.
    """

    x = np.asarray(features, dtype=np.float64)
    edges = np.asarray(edge_index, dtype=np.int64)
    assignment = quadrant_assignment(positions, supernodes)

    pooled = np.zeros((supernodes, x.shape[1]), dtype=np.float64)
    for group in range(supernodes):
        members = assignment == group
        if members.any():
            pooled[group] = x[members].mean(axis=0)

    adjacency = np.zeros((supernodes, supernodes), dtype=np.float64)
    if edges.size:
        left = assignment[edges[0]]
        right = assignment[edges[1]]
        for a, b in zip(left, right):
            if a != b:
                adjacency[a, b] += 1.0
                adjacency[b, a] += 1.0
    largest = adjacency.max()
    if largest > 0:
        adjacency = adjacency / largest
    return pooled.astype(np.float32), adjacency.astype(np.float32)


def adjacency_for_mode(adjacency: torch.Tensor, mode: str) -> torch.Tensor:
    """Return the entangling weights for one variant of the ablation."""

    if mode == "graph":
        return adjacency
    if mode == "complete":
        return torch.ones_like(adjacency) - torch.eye(
            adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype
        )
    if mode == "none":
        return torch.zeros_like(adjacency)
    raise ValueError(f"Unknown entangle mode {mode!r}; expected {ENTANGLE_MODES}")


# --------------------------------------------------------------------------
# the circuit
# --------------------------------------------------------------------------


class GraphStructuredCircuit(nn.Module):
    """Edge-conditioned variational circuit over ``k`` supernode qubits."""

    def __init__(
        self,
        supernodes: int = DEFAULT_SUPERNODES,
        layers: int = 2,
        entangle: str = "graph",
        freeze: bool = False,
    ) -> None:
        super().__init__()
        if entangle not in ENTANGLE_MODES:
            raise ValueError(f"entangle must be one of {ENTANGLE_MODES}")
        if supernodes < 2 or layers < 1:
            raise ValueError("supernodes >= 2 and layers >= 1 are required")
        try:
            import pennylane as qml
        except ImportError as error:
            raise RuntimeError(
                "PennyLane 0.45.1 is required for the graph-structured circuit"
            ) from error

        self.supernodes = int(supernodes)
        self.layers = int(layers)
        self.entangle = entangle
        self.pairs = [
            (i, j) for i in range(supernodes) for j in range(i + 1, supernodes)
        ]
        wires = tuple(range(supernodes))
        device = qml.device("default.qubit", wires=supernodes)
        pairs = self.pairs

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, rotations, couplings):
            # inputs packs the k encoding angles followed by the k*k adjacency
            # TorchLayer keeps the batch axis inside the qnode, so the packed
            # adjacency is reshaped per sample and the coupling angles are
            # broadcast rather than scalar.
            angles = inputs[..., :supernodes]
            weights = inputs[..., supernodes:].reshape(
                *inputs.shape[:-1], supernodes, supernodes
            )
            for layer in range(layers):
                qml.AngleEmbedding(angles, wires=wires, rotation="Y")
                for wire in wires:
                    qml.Rot(*rotations[layer, wire], wires=wire)
                for index, (left, right) in enumerate(pairs):
                    qml.CRZ(
                        couplings[layer, index] * weights[..., left, right],
                        wires=[left, right],
                    )
            return [qml.expval(qml.PauliZ(wire)) for wire in wires]

        shapes = {
            "rotations": (layers, supernodes, 3),
            "couplings": (layers, len(pairs)),
        }
        self.layer = qml.qnn.TorchLayer(
            circuit, shapes, init_method=torch.nn.init.normal_
        )
        if freeze:
            for parameter in self.layer.parameters():
                parameter.requires_grad_(False)
        self.frozen = bool(freeze)

    def train(self, mode: bool = True) -> "GraphStructuredCircuit":
        super().train(mode)
        if self.frozen:
            for parameter in self.layer.parameters():
                parameter.requires_grad_(False)
        return self

    def forward(
        self, angles: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.supernodes:
            raise ValueError(f"angles must be [batch, {self.supernodes}]")
        if adjacency.shape[-2:] != (self.supernodes, self.supernodes):
            raise ValueError(
                f"adjacency must be [batch, {self.supernodes}, {self.supernodes}]"
            )
        weights = adjacency_for_mode(adjacency, self.entangle)
        packed = torch.cat(
            [angles, weights.reshape(angles.shape[0], -1)], dim=1
        )
        return self.layer(packed).to(dtype=angles.dtype)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.layer.parameters())


class ClassicalGraphControl(nn.Module):
    """Message passing with the same adjacency and a matched parameter budget.

    The classical arm of the comparison. It sees exactly the same coarsened
    graph, so a difference between it and the quantum arm cannot be attributed
    to one of them having more information about the graph.
    """

    def __init__(self, supernodes: int = DEFAULT_SUPERNODES, layers: int = 2) -> None:
        super().__init__()
        self.supernodes = int(supernodes)
        self.layers = int(layers)
        self.self_weight = nn.Parameter(torch.randn(layers, supernodes, 3) * 0.1)
        pairs = supernodes * (supernodes - 1) // 2
        self.pair_weight = nn.Parameter(torch.randn(layers, pairs) * 0.1)
        self.pairs = [
            (i, j) for i in range(supernodes) for j in range(i + 1, supernodes)
        ]

    def forward(
        self, angles: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        state = torch.tanh(angles)
        for layer in range(self.layers):
            rotated = torch.tanh(
                state * self.self_weight[layer, :, 0]
                + self.self_weight[layer, :, 1]
            ) * self.self_weight[layer, :, 2]
            messages = torch.zeros_like(state)
            for index, (left, right) in enumerate(self.pairs):
                weight = self.pair_weight[layer, index] * adjacency[:, left, right]
                messages[:, left] = messages[:, left] + weight * state[:, right]
                messages[:, right] = messages[:, right] + weight * state[:, left]
            state = torch.tanh(rotated + messages)
        return state

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GraphQuantumHead(nn.Module):
    """Classifier over coarsened graphs, quantum or classical at the same budget."""

    VARIANTS = (
        "graph_quantum",
        "complete_quantum",
        "no_entangle",
        "classical_gnn",
        "random_fixed",
    )

    def __init__(
        self,
        labels: int,
        node_features: int,
        *,
        variant: str = "graph_quantum",
        supernodes: int = DEFAULT_SUPERNODES,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.variant = variant
        self.supernodes = int(supernodes)
        self.encoder = nn.Linear(node_features, 1)

        if variant == "classical_gnn":
            self.core = ClassicalGraphControl(supernodes, layers)
        else:
            mode = {
                "graph_quantum": "graph",
                "complete_quantum": "complete",
                "no_entangle": "none",
                "random_fixed": "graph",
            }[variant]
            self.core = GraphStructuredCircuit(
                supernodes, layers, entangle=mode, freeze=variant == "random_fixed"
            )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(supernodes, labels)
        )

    def forward(
        self, pooled: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        if pooled.ndim != 3 or pooled.shape[1] != self.supernodes:
            raise ValueError(f"pooled must be [batch, {self.supernodes}, features]")
        angles = np.pi * torch.tanh(self.encoder(pooled).squeeze(-1))
        return self.classifier(self.core(angles, adjacency))

    def budget(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "core_parameters": self.core.parameter_count(),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "optimiser_updated_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "circuit_frozen": self.variant == "random_fixed",
        }
