from __future__ import annotations

import unittest
from importlib.util import find_spec

import numpy as np

from cxr_thesis.objective3_v2.kernels import (
    angle_statevectors,
    bloch_features,
    classical_kernels,
    fidelity_kernel,
    geometric_difference,
    geometric_difference_sweep,
    interpret,
    iqp_statevectors,
    normalise_trace,
    projected_quantum_kernel,
    reduce_to_qubits,
    statevectors,
)

HAS_PENNYLANE = find_spec("pennylane") is not None

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def slow_iqp_state(features: np.ndarray, qubits: int, repeats: int) -> np.ndarray:
    """Explicit reference: build the phase basis state by basis state."""

    dimension = 2**qubits
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    full_hadamard = np.array([[1.0]], dtype=complex)
    for _ in range(qubits):
        full_hadamard = np.kron(full_hadamard, hadamard)

    phases = np.empty(dimension, dtype=complex)
    for index in range(dimension):
        bits = [(index >> (qubits - 1 - k)) & 1 for k in range(qubits)]
        signs = [1.0 - 2.0 * bit for bit in bits]
        total = sum(features[k] * signs[k] for k in range(qubits))
        for k in range(qubits):
            for m in range(k + 1, qubits):
                total += features[k] * features[m] * signs[k] * signs[m]
        phases[index] = np.exp(1j * total)

    state = np.zeros(dimension, dtype=complex)
    state[0] = 1.0
    for _ in range(repeats):
        state = full_hadamard @ state
        state = phases * state
    return state


def slow_reduced_density(state: np.ndarray, qubits: int, qubit: int) -> np.ndarray:
    """Explicit partial trace down to one qubit."""

    tensor = state.reshape([2] * qubits)
    others = [axis for axis in range(qubits) if axis != qubit]
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    return moved @ moved.conj().T


class StatevectorTests(unittest.TestCase):
    def test_states_are_normalised(self) -> None:
        generator = np.random.default_rng(0)
        features = generator.uniform(-np.pi, np.pi, size=(12, 4))
        for encoding in ("angle", "iqp"):
            states = statevectors(features, 4, encoding)
            norms = np.sum(np.abs(states) ** 2, axis=1)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-10), encoding)

    def test_iqp_matches_explicit_reference(self) -> None:
        generator = np.random.default_rng(1)
        features = generator.uniform(-np.pi, np.pi, size=(6, 3))
        fast = iqp_statevectors(features, 3, repeats=2)
        for row in range(features.shape[0]):
            slow = slow_iqp_state(features[row], 3, repeats=2)
            self.assertTrue(np.allclose(fast[row], slow, atol=1e-10))

    def test_angle_encoding_entangles(self) -> None:
        # With the CZ ring the state must not remain a product state.
        features = np.array([[1.0, 0.9, 1.2, 0.7]])
        state = angle_statevectors(features, 4, layers=2)
        density = slow_reduced_density(state[0], 4, 0)
        purity = float(np.real(np.trace(density @ density)))
        self.assertLess(purity, 0.999)

    def test_unknown_encoding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            statevectors(np.zeros((2, 3)), 3, "amplitude")

    def test_feature_shape_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            angle_statevectors(np.zeros((5, 3)), 4)

    @unittest.skipUnless(HAS_PENNYLANE, "pennylane is not installed")
    def test_angle_layer_matches_pennylane(self) -> None:
        import pennylane as qml

        qubits = 4
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device)
        def circuit(values):
            qml.AngleEmbedding(values, wires=range(qubits), rotation="Y")
            for wire in range(qubits):
                qml.CZ(wires=[wire, (wire + 1) % qubits])
            return qml.state()

        generator = np.random.default_rng(2)
        features = generator.uniform(-np.pi, np.pi, size=(5, qubits))
        mine = angle_statevectors(features, qubits, layers=1)
        for row in range(features.shape[0]):
            theirs = np.asarray(circuit(features[row]))
            self.assertTrue(np.allclose(mine[row], theirs, atol=1e-10))

    @unittest.skipUnless(HAS_PENNYLANE, "pennylane is not installed")
    def test_bloch_features_match_pennylane_expectations(self) -> None:
        import pennylane as qml

        qubits = 3
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device)
        def circuit(values):
            qml.AngleEmbedding(values, wires=range(qubits), rotation="Y")
            for wire in range(qubits):
                qml.CZ(wires=[wire, (wire + 1) % qubits])
            return (
                [qml.expval(qml.PauliX(w)) for w in range(qubits)]
                + [qml.expval(qml.PauliY(w)) for w in range(qubits)]
                + [qml.expval(qml.PauliZ(w)) for w in range(qubits)]
            )

        generator = np.random.default_rng(3)
        features = generator.uniform(-np.pi, np.pi, size=(4, qubits))
        states = angle_statevectors(features, qubits, layers=1)
        mine = bloch_features(states, qubits)
        for row in range(features.shape[0]):
            expected = np.asarray(circuit(features[row]), dtype=float)
            for qubit in range(qubits):
                self.assertAlmostEqual(mine[row, 3 * qubit + 0], expected[qubit], places=9)
                self.assertAlmostEqual(
                    mine[row, 3 * qubit + 1], expected[qubits + qubit], places=9
                )
                self.assertAlmostEqual(
                    mine[row, 3 * qubit + 2], expected[2 * qubits + qubit], places=9
                )


class KernelTests(unittest.TestCase):
    def test_trace_normalisation(self) -> None:
        generator = np.random.default_rng(4)
        base = generator.normal(size=(20, 6))
        kernel = normalise_trace(base @ base.T)
        self.assertAlmostEqual(float(np.trace(kernel)), 20.0, places=8)

    def test_fidelity_kernel_is_psd_with_unit_self_overlap(self) -> None:
        generator = np.random.default_rng(5)
        features = generator.uniform(-np.pi, np.pi, size=(25, 4))
        states = angle_statevectors(features, 4)
        raw = np.abs(states.conj() @ states.T) ** 2
        self.assertTrue(np.allclose(np.diag(raw), 1.0, atol=1e-10))
        eigenvalues = np.linalg.eigvalsh(fidelity_kernel(states))
        self.assertGreater(eigenvalues.min(), -1e-8)

    def test_projected_kernel_matches_explicit_density_matrices(self) -> None:
        generator = np.random.default_rng(6)
        qubits, gamma = 3, 0.7
        features = generator.uniform(-np.pi, np.pi, size=(8, qubits))
        states = iqp_statevectors(features, qubits)
        fast = projected_quantum_kernel(states, qubits, gamma=gamma)

        samples = states.shape[0]
        densities = [
            [slow_reduced_density(states[i], qubits, k) for k in range(qubits)]
            for i in range(samples)
        ]
        slow = np.empty((samples, samples))
        for i in range(samples):
            for j in range(samples):
                total = sum(
                    float(np.linalg.norm(densities[i][k] - densities[j][k], "fro") ** 2)
                    for k in range(qubits)
                )
                slow[i, j] = np.exp(-gamma * total)
        self.assertTrue(np.allclose(fast, normalise_trace(slow), atol=1e-9))

    def test_classical_family_is_normalised_and_named(self) -> None:
        generator = np.random.default_rng(7)
        values = generator.normal(size=(30, 12))
        kernels = classical_kernels(values)
        self.assertIn("linear", kernels)
        self.assertEqual(len(kernels), 4)
        for name, kernel in kernels.items():
            self.assertAlmostEqual(float(np.trace(kernel)), 30.0, places=6, msg=name)

    def test_reduce_to_qubits_bounds_angles(self) -> None:
        generator = np.random.default_rng(8)
        values = generator.normal(size=(40, 160))
        reduced = reduce_to_qubits(values, 6)
        self.assertEqual(reduced.shape, (40, 6))
        self.assertLessEqual(float(np.max(np.abs(reduced))), np.pi + 1e-9)


class GeometricDifferenceTests(unittest.TestCase):
    def _kernel(self, seed: int, samples: int = 40) -> np.ndarray:
        generator = np.random.default_rng(seed)
        base = generator.normal(size=(samples, 8))
        return normalise_trace(base @ base.T + 0.5 * np.eye(samples))

    def test_identical_kernels_give_unit_difference(self) -> None:
        kernel = self._kernel(9)
        result = geometric_difference(kernel, kernel, regularisation=1e-10)
        self.assertAlmostEqual(result.geometric_difference, 1.0, places=4)

    def test_ratio_and_flag_are_consistent(self) -> None:
        kernel = self._kernel(10)
        result = geometric_difference(kernel, kernel, regularisation=1e-10)
        self.assertAlmostEqual(
            result.ratio, result.geometric_difference / np.sqrt(result.samples)
        )
        self.assertEqual(result.advantage_possible, result.ratio >= 0.5)

    def test_scaling_the_quantum_kernel_scales_the_difference(self) -> None:
        # g is homogeneous of degree 1/2 in K_Q, so a 4x kernel doubles g.
        kernel = self._kernel(11)
        single = geometric_difference(kernel, kernel, regularisation=1e-10)
        quadruple = geometric_difference(kernel, 4.0 * kernel, regularisation=1e-10)
        self.assertAlmostEqual(
            quadruple.geometric_difference / single.geometric_difference, 2.0, places=3
        )

    def test_unrelated_kernels_separate_more_than_identical_ones(self) -> None:
        classical = self._kernel(12)
        quantum = self._kernel(13)
        same = geometric_difference(classical, classical, regularisation=1e-6)
        other = geometric_difference(classical, quantum, regularisation=1e-6)
        self.assertGreater(other.geometric_difference, same.geometric_difference)

    def test_regularisation_sweep_is_monotone_decreasing(self) -> None:
        classical = self._kernel(14)
        quantum = self._kernel(15)
        sweep = geometric_difference_sweep(classical, quantum)
        values = [row["geometric_difference"] for row in sweep]
        self.assertEqual(len(values), 4)
        for earlier, later in zip(values, values[1:]):
            self.assertGreaterEqual(earlier + 1e-9, later)

    def test_mismatched_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            geometric_difference(np.eye(4), np.eye(5))

    def test_non_positive_regularisation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            geometric_difference(np.eye(4), np.eye(4), regularisation=0.0)

    def test_interpretation_states_the_logical_direction(self) -> None:
        ruled_out = interpret([0.02, 0.05, 0.11])
        self.assertIn("no separation", ruled_out.lower())
        self.assertIn("predicted outcome", ruled_out)
        possible = interpret([0.02, 0.71])
        self.assertIn("necessary but not a sufficient", possible)


if __name__ == "__main__":
    unittest.main()
