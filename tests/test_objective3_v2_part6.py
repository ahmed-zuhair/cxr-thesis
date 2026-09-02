from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import torch

from cxr_thesis.objective3_v2.graph_quantum import (
    ENTANGLE_MODES,
    ClassicalGraphControl,
    GraphQuantumHead,
    GraphStructuredCircuit,
    adjacency_for_mode,
    coarsen,
    quadrant_assignment,
)

HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None


def grid_graph(rows: int = 4, columns: int = 4, features: int = 7):
    """A lattice patch graph: positions on a grid, edges between neighbours."""

    positions = np.array(
        [[r, c] for r in range(rows) for c in range(columns)], dtype=float
    )
    generator = np.random.default_rng(0)
    x = generator.normal(size=(rows * columns, features))
    edges = []
    for r in range(rows):
        for c in range(columns):
            node = r * columns + c
            if c + 1 < columns:
                edges.append((node, node + 1))
            if r + 1 < rows:
                edges.append((node, node + columns))
    return x, np.array(edges).T, positions


class CoarseningTests(unittest.TestCase):
    def test_quadrants_partition_every_node(self) -> None:
        _, _, positions = grid_graph()
        assignment = quadrant_assignment(positions, 4)
        self.assertEqual(assignment.shape, (positions.shape[0],))
        self.assertTrue(set(np.unique(assignment)).issubset({0, 1, 2, 3}))

    def test_two_supernodes_split_on_one_axis(self) -> None:
        _, _, positions = grid_graph()
        assignment = quadrant_assignment(positions, 2)
        self.assertTrue(set(np.unique(assignment)).issubset({0, 1}))

    def test_adjacency_is_symmetric_with_zero_diagonal(self) -> None:
        x, edges, positions = grid_graph()
        _, adjacency = coarsen(x, edges, positions, 4)
        self.assertTrue(np.allclose(adjacency, adjacency.T))
        self.assertTrue(np.allclose(np.diag(adjacency), 0.0))
        self.assertLessEqual(adjacency.max(), 1.0 + 1e-6)

    def test_pooled_features_are_group_means(self) -> None:
        x, edges, positions = grid_graph()
        pooled, _ = coarsen(x, edges, positions, 4)
        assignment = quadrant_assignment(positions, 4)
        for group in range(4):
            members = assignment == group
            if members.any():
                self.assertTrue(
                    np.allclose(pooled[group], x[members].mean(axis=0), atol=1e-5)
                )

    def test_a_graph_with_no_edges_gives_zero_adjacency(self) -> None:
        x, _, positions = grid_graph()
        _, adjacency = coarsen(x, np.zeros((2, 0), dtype=int), positions, 4)
        self.assertTrue(np.allclose(adjacency, 0.0))

    def test_bad_positions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quadrant_assignment(np.zeros((5, 1)), 4)
        with self.assertRaises(ValueError):
            quadrant_assignment(np.zeros((5, 2)), 3)


class ModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adjacency = torch.tensor(
            [[[0.0, 0.5, 0.0, 1.0],
              [0.5, 0.0, 0.2, 0.0],
              [0.0, 0.2, 0.0, 0.0],
              [1.0, 0.0, 0.0, 0.0]]]
        )

    def test_graph_mode_is_the_real_adjacency(self) -> None:
        self.assertTrue(
            torch.allclose(adjacency_for_mode(self.adjacency, "graph"), self.adjacency)
        )

    def test_complete_mode_is_all_pairs_without_self_loops(self) -> None:
        weights = adjacency_for_mode(self.adjacency, "complete")
        self.assertTrue(torch.allclose(torch.diagonal(weights, dim1=-2, dim2=-1),
                                       torch.zeros(4)))
        off = weights[0][~torch.eye(4, dtype=bool)]
        self.assertTrue(torch.allclose(off, torch.ones_like(off)))

    def test_none_mode_removes_every_coupling(self) -> None:
        self.assertTrue(
            torch.allclose(
                adjacency_for_mode(self.adjacency, "none"),
                torch.zeros_like(self.adjacency),
            )
        )

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            adjacency_for_mode(self.adjacency, "spanning_tree")


@unittest.skipUnless(HAS_PENNYLANE, "pennylane is not installed")
class CircuitTests(unittest.TestCase):
    def test_every_variant_is_parameter_matched(self) -> None:
        budgets = {}
        for variant in GraphQuantumHead.VARIANTS:
            head = GraphQuantumHead(12, 7, variant=variant)
            budgets[variant] = head.budget()
        cores = {b["core_parameters"] for b in budgets.values()}
        totals = {b["total_parameters"] for b in budgets.values()}
        self.assertEqual(len(cores), 1, f"core budgets differ: {budgets}")
        self.assertEqual(len(totals), 1, f"total budgets differ: {budgets}")
        # only the frozen control trains fewer
        self.assertLess(
            budgets["random_fixed"]["optimiser_updated_parameters"],
            budgets["graph_quantum"]["optimiser_updated_parameters"],
        )

    def test_zero_adjacency_makes_graph_and_none_identical(self) -> None:
        # CRZ(0) is the identity, so an edgeless graph must equal the
        # no-entangle variant exactly. This is what licenses reading the
        # comparison as "topology or not" rather than "gates or not".
        torch.manual_seed(0)
        circuit = GraphStructuredCircuit(4, 2, entangle="graph")
        angles = torch.randn(3, 4)
        zeros = torch.zeros(3, 4, 4)
        with torch.no_grad():
            structured = circuit(angles, zeros)
            circuit.entangle = "none"
            removed = circuit(angles, zeros)
        self.assertTrue(torch.allclose(structured, removed, atol=1e-6))

    def test_adjacency_changes_the_output(self) -> None:
        torch.manual_seed(0)
        circuit = GraphStructuredCircuit(4, 2, entangle="graph")
        angles = torch.randn(2, 4)
        empty = torch.zeros(2, 4, 4)
        dense = torch.ones(2, 4, 4) - torch.eye(4)
        with torch.no_grad():
            self.assertFalse(
                torch.allclose(circuit(angles, empty), circuit(angles, dense), atol=1e-4)
            )

    def test_frozen_variant_receives_no_gradient(self) -> None:
        head = GraphQuantumHead(12, 7, variant="random_fixed")
        head.train()
        head(torch.randn(4, 4, 7), torch.rand(4, 4, 4)).sum().backward()
        for parameter in head.core.parameters():
            self.assertIsNone(parameter.grad)
        self.assertIsNotNone(head.encoder.weight.grad)

    def test_shapes_are_validated(self) -> None:
        head = GraphQuantumHead(12, 7)
        with self.assertRaises(ValueError):
            head(torch.randn(4, 3, 7), torch.rand(4, 4, 4))
        with self.assertRaises(ValueError):
            GraphStructuredCircuit(4, 2)(torch.randn(2, 3), torch.rand(2, 4, 4))

    def test_topology_signal_is_detectable(self) -> None:
        """The decisive comparison must be able to see a real graph signal.

        A test that only checks the code runs cannot distinguish a working
        ablation from a silently broken one. Here the label is a function of
        the adjacency alone, with node features carrying no information at all,
        so a circuit that reads the topology must beat one that ignores it.
        """

        generator = np.random.default_rng(3)
        cases = 200
        signal = generator.integers(0, 2, size=cases)
        pooled = torch.tensor(
            generator.normal(size=(cases, 4, 7)).astype(np.float32)
        )
        adjacency = torch.zeros(cases, 4, 4)
        for index in range(cases):
            weight = 1.0 if signal[index] else 0.0
            adjacency[index, 0, 1] = adjacency[index, 1, 0] = weight
            adjacency[index, 2, 3] = adjacency[index, 3, 2] = weight
        labels = torch.tensor(signal.astype(np.float32))

        def fit(variant: str) -> float:
            torch.manual_seed(0)
            head = GraphQuantumHead(1, 7, variant=variant)
            optimiser = torch.optim.Adam(
                [p for p in head.parameters() if p.requires_grad], lr=0.05
            )
            criterion = torch.nn.BCEWithLogitsLoss()
            for _ in range(60):
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(
                    head(pooled, adjacency).squeeze(-1), labels
                )
                loss.backward()
                optimiser.step()
            with torch.no_grad():
                scores = head(pooled, adjacency).squeeze(-1)
            order = torch.argsort(scores)
            ranks = torch.empty_like(order, dtype=torch.float64)
            ranks[order] = torch.arange(len(order), dtype=torch.float64)
            positive = labels == 1
            n_pos = int(positive.sum())
            n_neg = len(labels) - n_pos
            return float(
                (ranks[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
            )

        structured = fit("graph_quantum")
        blind = fit("no_entangle")
        self.assertGreater(structured, 0.7, "structured circuit failed to learn")
        self.assertGreater(
            structured - blind, 0.1, "the ablation cannot see a pure topology signal"
        )


class ClassicalControlTests(unittest.TestCase):
    def test_control_matches_the_quantum_budget(self) -> None:
        control = ClassicalGraphControl(4, 2)
        self.assertEqual(control.parameter_count(), 4 * 2 * 3 + 2 * 6)

    def test_control_uses_the_adjacency(self) -> None:
        torch.manual_seed(0)
        control = ClassicalGraphControl(4, 2)
        angles = torch.randn(2, 4)
        with torch.no_grad():
            empty = control(angles, torch.zeros(2, 4, 4))
            dense = control(angles, torch.ones(2, 4, 4))
        self.assertFalse(torch.allclose(empty, dense, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
