from __future__ import annotations

import subprocess
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

import torch

from cxr_thesis.objective2.data import GraphBatch
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective3.models import (
    ClassicalMatchedBottleneck,
    HybridGraphHead,
    QuantumBottleneck,
    bottleneck_parameter_count,
)


class Objective3ArchitectureTests(unittest.TestCase):
    def test_graph_classifier_exposes_fused_embedding(self) -> None:
        batch = GraphBatch(
            x=torch.rand(8, 7),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
            ),
            batch_index=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            clinical=torch.rand(2, 9),
            labels=torch.randint(0, 2, (2, 12)).float(),
        )
        model = build_classifier("gat", 12, node_dim=7)
        embedding = model.encode(batch)
        self.assertEqual(tuple(embedding.shape), (2, 160))
        self.assertEqual(tuple(model(batch).shape), (2, 12))

    def test_classical_control_has_frozen_parameter_budget(self) -> None:
        bottleneck = ClassicalMatchedBottleneck()
        self.assertEqual(bottleneck_parameter_count(bottleneck), 24)
        output = bottleneck(torch.rand(3, 4))
        self.assertEqual(tuple(output.shape), (3, 4))
        output.mean().backward()

    def test_hybrid_classical_head_shape(self) -> None:
        model = HybridGraphHead(12, bottleneck="classical_matched")
        output = model(torch.rand(3, 160))
        self.assertEqual(tuple(output.shape), (3, 12))

    @unittest.skipUnless(find_spec("pennylane"), "PennyLane is optional")
    def test_quantum_control_matches_classical_parameter_budget(self) -> None:
        classical = ClassicalMatchedBottleneck()
        quantum = QuantumBottleneck()
        self.assertEqual(bottleneck_parameter_count(classical), 24)
        self.assertEqual(bottleneck_parameter_count(quantum), 24)
        inputs = torch.rand(3, 4, requires_grad=True)
        output = quantum(inputs)
        self.assertEqual(tuple(output.shape), (3, 4))
        output.mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all().item())

    def test_quantum_smoke_cli_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts" / "smoke_objective3_quantum.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--batch-size", result.stdout)


if __name__ == "__main__":
    unittest.main()
