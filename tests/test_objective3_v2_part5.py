from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cxr_thesis.objective3.models import (
    EnhancedHybridGraphHead,
    QuantumRandomBottleneck,
    bottleneck_parameter_count,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "learning_curve", REPOSITORY_ROOT / "scripts" / "run_objective3_v2_learning_curve.py"
)
learning_curve = importlib.util.module_from_spec(_spec)
sys.modules["learning_curve"] = learning_curve
_spec.loader.exec_module(learning_curve)

HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None


def _manifests(directory: Path, train_rows: int = 400) -> tuple[Path, Path]:
    train = pd.DataFrame(
        {
            "image_id": [f"img{i:05d}" for i in range(train_rows)],
            "patient_id": [f"p{i // 2:05d}" for i in range(train_rows)],
        }
    )
    validation = pd.DataFrame(
        {
            "image_id": [f"v{i:05d}" for i in range(50)],
            "patient_id": [f"vp{i:05d}" for i in range(50)],
        }
    )
    train_path, val_path = directory / "train.csv", directory / "val.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(val_path, index=False)
    return train_path, val_path


@unittest.skipUnless(HAS_PENNYLANE, "pennylane is not installed")
class FrozenControlTests(unittest.TestCase):
    def test_circuit_angles_are_frozen(self) -> None:
        bottleneck = QuantumRandomBottleneck()
        self.assertTrue(
            all(not p.requires_grad for p in bottleneck.parameters())
        )
        # capacity is unchanged: the control is parameter-matched, not smaller
        self.assertEqual(bottleneck_parameter_count(bottleneck), 36)

    def test_freeze_survives_train_mode(self) -> None:
        # nn.Module.train() must not silently re-enable the circuit
        bottleneck = QuantumRandomBottleneck()
        bottleneck.train()
        self.assertTrue(all(not p.requires_grad for p in bottleneck.parameters()))

    def test_head_budget_matches_the_other_arms(self) -> None:
        counts = {}
        for name in ("quantum", "classical_matched", "quantum_random"):
            head = EnhancedHybridGraphHead(12, bottleneck=name)
            counts[name] = (
                bottleneck_parameter_count(head.bottleneck),
                sum(p.numel() for p in head.parameters()),
                sum(p.numel() for p in head.parameters() if p.requires_grad),
            )
        self.assertEqual(counts["quantum"][:2], (36, 3253))
        self.assertEqual(counts["classical_matched"][:2], (36, 3253))
        self.assertEqual(counts["quantum_random"][:2], (36, 3253))
        # only the frozen arm trains fewer parameters
        self.assertEqual(counts["quantum"][2], 3253)
        self.assertEqual(counts["quantum_random"][2], 3217)

    def test_frozen_circuit_receives_no_gradient(self) -> None:
        head = EnhancedHybridGraphHead(12, bottleneck="quantum_random")
        head.train()
        head(torch.randn(4, 160)).sum().backward()
        for parameter in head.bottleneck.parameters():
            self.assertIsNone(parameter.grad)
        self.assertIsNotNone(head.input_projection.weight.grad)

    def test_unknown_bottleneck_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            EnhancedHybridGraphHead(12, bottleneck="quantum_magic")


class PrefixAuditTests(unittest.TestCase):
    def test_audit_reports_a_usable_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train, validation = _manifests(Path(directory))
            audit = learning_curve.prefix_audit(train, validation, [50, 100, 200])
            self.assertTrue(audit["nested_by_construction"])
            self.assertEqual(audit["patient_overlap_train_validation"], 0)
            self.assertTrue(audit["validation_fixed_at_full_size"])
            self.assertIn("limit-train", audit["subset_mechanism"])

    def test_prefixes_really_are_nested(self) -> None:
        # The whole point of using --limit-train rather than a resampled subset:
        # a prefix of a fixed order is inside every longer prefix, always.
        rows = list(range(400))
        for smaller, larger in ((50, 100), (100, 200)):
            self.assertTrue(set(rows[:smaller]).issubset(rows[:larger]))

    def test_patient_overlap_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation = _manifests(root)
            leaked = pd.read_csv(validation)
            leaked.loc[0, "patient_id"] = "p00000"  # a training patient
            leaked.to_csv(validation, index=False)
            with self.assertRaises(RuntimeError):
                learning_curve.prefix_audit(train, validation, [50])

    def test_size_larger_than_cohort_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train, validation = _manifests(Path(directory), train_rows=100)
            with self.assertRaises(ValueError):
                learning_curve.prefix_audit(train, validation, [500])

    def test_missing_patient_column_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation = _manifests(root)
            frame = pd.read_csv(train).drop(columns=["patient_id"])
            frame.to_csv(train, index=False)
            with self.assertRaises(ValueError):
                learning_curve.prefix_audit(train, validation, [50])


class VerdictTests(unittest.TestCase):
    def _runs(self, advantage: float, sizes=(100, 500, 30000)) -> tuple[list, list]:
        seeds = list(range(10))
        runs = []
        generator = np.random.default_rng(1)
        for size in sizes:
            for variant in learning_curve.VARIANTS:
                bump = advantage if (variant == "quantum" and size <= 1000) else 0.0
                for seed in seeds:
                    runs.append(
                        {
                            "n_train": size,
                            "variant": variant,
                            "seed": seed,
                            "validation_macro_auroc": 0.65
                            + bump
                            + generator.normal(0, 0.0002),
                        }
                    )
        return runs, seeds

    def test_verdict_passes_on_a_real_small_data_advantage(self) -> None:
        runs, seeds = self._runs(advantage=0.02)
        analysis = learning_curve.curve_and_deltas(runs, seeds)
        verdict = learning_curve.h2_verdict(analysis["deltas"])
        self.assertTrue(verdict["passed"])
        self.assertTrue(all(s <= 1000 for s in verdict["passing_sizes"]))

    def test_verdict_fails_when_there_is_no_advantage(self) -> None:
        runs, seeds = self._runs(advantage=0.0)
        analysis = learning_curve.curve_and_deltas(runs, seeds)
        self.assertFalse(learning_curve.h2_verdict(analysis["deltas"])["passed"])

    def test_advantage_below_threshold_does_not_pass(self) -> None:
        # 0.002 is a real shift but under the preregistered 0.005 threshold
        runs, seeds = self._runs(advantage=0.002)
        analysis = learning_curve.curve_and_deltas(runs, seeds)
        self.assertFalse(learning_curve.h2_verdict(analysis["deltas"])["passed"])

    def test_large_sample_advantage_does_not_satisfy_h2(self) -> None:
        # H2 is specifically about the small-data regime
        seeds = list(range(10))
        runs = []
        for variant in learning_curve.VARIANTS:
            bump = 0.02 if variant == "quantum" else 0.0
            for seed in seeds:
                runs.append(
                    {
                        "n_train": 30000,
                        "variant": variant,
                        "seed": seed,
                        "validation_macro_auroc": 0.65 + bump,
                    }
                )
        analysis = learning_curve.curve_and_deltas(runs, seeds)
        self.assertFalse(learning_curve.h2_verdict(analysis["deltas"])["passed"])

    def test_both_controls_are_compared(self) -> None:
        runs, seeds = self._runs(advantage=0.01)
        analysis = learning_curve.curve_and_deltas(runs, seeds)
        comparisons = {row["comparison"] for row in analysis["deltas"]}
        self.assertEqual(
            comparisons,
            {"quantum_minus_classical_matched", "quantum_minus_quantum_random"},
        )

    def test_bootstrap_p_is_never_reported_as_zero(self) -> None:
        runs, seeds = self._runs(advantage=0.01)
        analysis = learning_curve.curve_and_deltas(runs, seeds)
        for row in analysis["deltas"]:
            self.assertTrue(row["bootstrap_p_report"].startswith("p < "))


class DriftTests(unittest.TestCase):
    def _summary(self, **overrides) -> dict:
        summary = {
            "architecture_version": "v1_1_reupload_gated",
            "variant": "quantum",
            "seed": 42,
            "test_cases_accessed": 0,
            "test_evaluated": False,
            "bottleneck_parameters": 36,
            "total_trainable_parameters": 3253,
        }
        summary.update(overrides)
        return summary

    def test_clean_summary_passes(self) -> None:
        learning_curve.check_summary(self._summary(), "quantum", 42)

    def test_wrong_architecture_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            learning_curve.check_summary(
                self._summary(architecture_version="v1_concat"), "quantum", 42
            )

    def test_test_access_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            learning_curve.check_summary(
                self._summary(test_cases_accessed=5), "quantum", 42
            )

    def test_random_arm_must_declare_a_frozen_circuit(self) -> None:
        with self.assertRaises(RuntimeError):
            learning_curve.check_summary(
                self._summary(variant="quantum_random"), "quantum_random", 42
            )
        learning_curve.check_summary(
            self._summary(variant="quantum_random", circuit_frozen=True),
            "quantum_random",
            42,
        )


if __name__ == "__main__":
    unittest.main()
