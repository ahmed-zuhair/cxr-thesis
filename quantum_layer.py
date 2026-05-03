"""
quantum_layer.py
================
Variational quantum circuit as a PyTorch layer.

Design:
    Four qubits with angle encoding and a strongly entangling ansatz.
    Inputs: four classical features per sample.
    Outputs: four expectation values from Pauli-Z measurements.

The strongly entangling layer alternates parameterised single-qubit
rotations with CNOT entangling gates in a fixed pattern. Two such layers
provide reasonable expressivity while keeping simulation cost low.

References:
    Schuld et al. 2020 — Circuit-centric quantum classifiers
    PennyLane documentation — qml.StronglyEntanglingLayers
"""
import torch
import torch.nn as nn
import pennylane as qml


N_QUBITS = 4
N_LAYERS = 2


def make_quantum_node(n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
    """
    Build a PennyLane QNode that takes inputs and trainable weights and
    returns expectation values from Pauli-Z measurements on each qubit.

    We use default.qubit because it supports the backprop differentiation
    method, which gives the fastest gradient computation for small circuits.
    For four qubits, default.qubit is fast enough that the lightning.qubit
    speedup is unnecessary.
    """
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Angle encoding: each input feature becomes a rotation angle on a qubit.
        # The four classical inputs map directly onto the four qubits.
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # Strongly entangling layers: trainable rotations with CNOT entanglement.
        # weights shape must be (n_layers, n_qubits, 3) for this template.
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        # Measure each qubit in the computational basis (Pauli-Z expectation).
        # Each output is in [-1, 1], suitable as features for downstream layers.
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class QuantumLayer(nn.Module):
    """
    PyTorch module wrapper around the PennyLane quantum circuit.

    Input:  (batch, n_qubits) tensor of classical features
    Output: (batch, n_qubits) tensor of quantum measurement outputs

    The circuit weights are trainable PyTorch parameters, optimised through
    automatic differentiation using PennyLane's torch interface.
    """
    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit = make_quantum_node(n_qubits, n_layers)

        # Trainable circuit parameters. Shape is (n_layers, n_qubits, 3).
        # The 3 corresponds to the three Euler angles per qubit per layer.
        weight_shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(
            torch.empty(weight_shape).uniform_(-0.1, 0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The PennyLane circuit processes one sample at a time.
        # We loop over the batch and stack the results.
        # For four qubits this is fast enough; for larger circuits a vectorised
        # backend like default.qubit.torch would be more efficient.
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            single_output = self.circuit(x[i], self.weights)
            # The circuit returns a list of tensors; stack them into one tensor.
            single_output = torch.stack(single_output)
            outputs.append(single_output)
        return torch.stack(outputs)


if __name__ == "__main__":
    # Sanity check
    layer = QuantumLayer(n_qubits=4, n_layers=2)
    print(f"Trainable parameters: {sum(p.numel() for p in layer.parameters())}")
    print(f"Weight tensor shape: {tuple(layer.weights.shape)}")

    x = torch.randn(8, 4)
    y = layer(x)
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")

    loss = y.sum()
    loss.backward()
    print(f"Gradient shape on weights: {tuple(layer.weights.grad.shape)}")
    print("Quantum layer end-to-end works")
