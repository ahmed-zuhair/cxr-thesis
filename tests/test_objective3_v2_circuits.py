from __future__ import annotations

import unittest
from importlib.util import find_spec

import numpy as np

from cxr_thesis.objective3_v2.circuits import (
    ANSATZE,
    barren_plateau_slope,
    encoding_expressibility,
    entangling_capability,
    expressibility,
    fidelity_kl_divergence,
    gradient_variance,
    haar_bin_probabilities,
    meyer_wallach,
    rot_matrices,
    strongly_entangling,
    v1_0_states,
    v1_1_states,
)

HAS_PENNYLANE = find_spec("pennylane") is not None


class GateTests(unittest.TestCase):
    def test_rot_is_unitary(self) -> None:
        generator = np.random.default_rng(0)
        angles = generator.uniform(-np.pi, np.pi, size=(20, 3))
        matrices = rot_matrices(angles[:, 0], angles[:, 1], angles[:, 2])
        for matrix in matrices:
            self.assertTrue(
                np.allclose(matrix @ matrix.conj().T, np.eye(2), atol=1e-12)
            )

    def test_ansatz_states_are_normalised(self) -> None:
        generator = np.random.default_rng(1)
        for name, (builder, layers, _) in ANSATZE.items():
            weights = generator.uniform(-np.pi, np.pi, size=(8, layers, 4, 3))
            inputs = generator.uniform(-np.pi, np.pi, size=(8, 4))
            states = builder(inputs, weights, 4)
            norms = np.sum(np.abs(states) ** 2, axis=1)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-10), name)

    def test_parameter_counts_match_the_published_models(self) -> None:
        # v1.0 published 24 bottleneck parameters, v1.1 published 36.
        self.assertEqual(ANSATZE["v1_0_bottleneck"][1] * 4 * 3, 24)
        self.assertEqual(ANSATZE["v1_1_reupload"][1] * 4 * 3, 36)
        self.assertEqual(ANSATZE["v1_0_bottleneck"][2], 24)
        self.assertEqual(ANSATZE["v1_1_reupload"][2], 36)

    @unittest.skipUnless(HAS_PENNYLANE, "pennylane is not installed")
    def test_v1_0_matches_pennylane(self) -> None:
        import pennylane as qml

        qubits, layers = 4, 2
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(qubits))
            return qml.state()

        generator = np.random.default_rng(2)
        for _ in range(4):
            inputs = generator.uniform(-np.pi, np.pi, size=qubits)
            weights = generator.uniform(-np.pi, np.pi, size=(layers, qubits, 3))
            mine = v1_0_states(inputs[None, :], weights[None, ...], qubits)[0]
            theirs = np.asarray(circuit(inputs, weights))
            self.assertTrue(np.allclose(mine, theirs, atol=1e-10))

    @unittest.skipUnless(HAS_PENNYLANE, "pennylane is not installed")
    def test_v1_1_reupload_matches_pennylane(self) -> None:
        import pennylane as qml

        qubits, layers = 4, 3
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device)
        def circuit(inputs, weights):
            for block in range(layers):
                qml.AngleEmbedding(inputs, wires=range(qubits), rotation="Y")
                qml.StronglyEntanglingLayers(
                    weights[block : block + 1], wires=range(qubits)
                )
            return qml.state()

        generator = np.random.default_rng(3)
        for _ in range(4):
            inputs = generator.uniform(-np.pi, np.pi, size=qubits)
            weights = generator.uniform(-np.pi, np.pi, size=(layers, qubits, 3))
            mine = v1_1_states(inputs[None, :], weights[None, ...], qubits)[0]
            theirs = np.asarray(circuit(inputs, weights))
            self.assertTrue(np.allclose(mine, theirs, atol=1e-10))


class HaarTests(unittest.TestCase):
    def test_bin_probabilities_sum_to_one(self) -> None:
        edges = np.linspace(0.0, 1.0, 76)
        for qubits in (2, 4, 6):
            self.assertAlmostEqual(
                float(haar_bin_probabilities(edges, qubits).sum()), 1.0, places=10
            )

    def test_haar_samples_score_near_zero_kl(self) -> None:
        # Fidelities drawn from the Haar law must look Haar-random.
        qubits = 4
        dimension = 2**qubits
        generator = np.random.default_rng(4)
        uniform = generator.uniform(size=40_000)
        fidelities = 1.0 - np.power(1.0 - uniform, 1.0 / (dimension - 1))
        self.assertLess(fidelity_kl_divergence(fidelities, qubits), 0.02)

    def test_degenerate_distribution_scores_high_kl(self) -> None:
        constant = np.full(5000, 0.99)
        self.assertGreater(fidelity_kl_divergence(constant, 4), 1.0)


class EntanglementTests(unittest.TestCase):
    def test_product_state_has_zero_measure(self) -> None:
        state = np.zeros((1, 16), dtype=complex)
        state[0, 0] = 1.0
        self.assertAlmostEqual(float(meyer_wallach(state, 4)[0]), 0.0, places=12)

    def test_bell_state_is_maximally_entangled(self) -> None:
        state = np.zeros((1, 4), dtype=complex)
        state[0, 0] = state[0, 3] = 1.0 / np.sqrt(2.0)
        self.assertAlmostEqual(float(meyer_wallach(state, 2)[0]), 1.0, places=12)

    def test_ghz_state_is_maximally_entangled(self) -> None:
        state = np.zeros((1, 8), dtype=complex)
        state[0, 0] = state[0, 7] = 1.0 / np.sqrt(2.0)
        self.assertAlmostEqual(float(meyer_wallach(state, 3)[0]), 1.0, places=12)

    def test_entangling_layers_actually_entangle(self) -> None:
        generator = np.random.default_rng(5)
        weights = generator.uniform(-np.pi, np.pi, size=(64, 2, 4, 3))
        state = np.zeros((64, 16), dtype=complex)
        state[:, 0] = 1.0
        measure = meyer_wallach(strongly_entangling(state, weights, 4), 4)
        self.assertGreater(float(np.mean(measure)), 0.3)


class DiagnosticTests(unittest.TestCase):
    def test_expressibility_runs_for_both_ansatze(self) -> None:
        for name in ANSATZE:
            result = expressibility(name, qubits=4, samples=400, seed=6)
            self.assertGreaterEqual(result["kl_divergence_from_haar"], 0.0)
            self.assertEqual(result["ansatz"], name)

    def test_deeper_circuits_are_more_expressible(self) -> None:
        shallow = expressibility("v1_0_bottleneck", layers=1, samples=3000, seed=7)
        deep = expressibility("v1_0_bottleneck", layers=6, samples=3000, seed=7)
        self.assertGreater(
            shallow["kl_divergence_from_haar"], deep["kl_divergence_from_haar"]
        )

    def test_entangling_capability_is_bounded(self) -> None:
        result = entangling_capability("v1_1_reupload", samples=400, seed=8)
        self.assertGreaterEqual(result["meyer_wallach_mean"], 0.0)
        self.assertLessEqual(result["meyer_wallach_mean"], 1.0)

    def test_gradient_variance_is_positive_at_small_scale(self) -> None:
        result = gradient_variance("v1_1_reupload", qubits=4, layers=3, samples=200)
        self.assertGreater(result["gradient_variance"], 0.0)

    def test_gradient_variance_falls_as_qubits_grow(self) -> None:
        variances = [
            gradient_variance(
                "v1_0_bottleneck", qubits=n, layers=6, samples=300, seed=9
            )["gradient_variance"]
            for n in (2, 4, 6, 8)
        ]
        self.assertGreater(variances[0], variances[-1])

    def test_bad_parameter_index_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gradient_variance("v1_1_reupload", qubits=4, layers=3, parameter=(9, 0, 0))

    def test_unknown_ansatz_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expressibility("not_an_ansatz")


class SlopeTests(unittest.TestCase):
    def test_exponential_decay_is_flagged_barren(self) -> None:
        counts = [2, 4, 6, 8, 10]
        variances = [2.0 ** (-n) for n in counts]
        fit = barren_plateau_slope(counts, variances)
        self.assertTrue(fit["barren"])
        self.assertAlmostEqual(fit["decay_base_per_qubit"], 0.5, places=6)

    def test_flat_variance_is_not_barren(self) -> None:
        fit = barren_plateau_slope([2, 4, 6, 8], [0.1, 0.1, 0.1, 0.1])
        self.assertFalse(fit["barren"])

    def test_insufficient_data_returns_none(self) -> None:
        self.assertIsNone(barren_plateau_slope([2], [0.0])["barren"])


class EncodingTests(unittest.TestCase):
    def test_both_encodings_report_a_measure(self) -> None:
        for encoding in ("angle", "iqp"):
            result = encoding_expressibility(encoding, qubits=4, samples=600, seed=10)
            self.assertGreaterEqual(result["kl_divergence_from_haar"], 0.0)
            self.assertGreaterEqual(result["meyer_wallach_mean"], 0.0)
            self.assertEqual(result["encoding"], encoding)


if __name__ == "__main__":
    unittest.main()
