from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cxr_thesis.objective3_v2.guards import (
    LockedTestAccessError,
    LockedTestAuthorisation,
    open_locked_test,
)
from cxr_thesis.objective3_v2.io_utils import write_json_atomic

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPOSITORY_ROOT / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


noise = _load("noise_resources", "run_objective3_v2_noise_resources.py")
final = _load("final_evaluation", "run_objective3_v2_final_evaluation.py")

HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None


class AuthorisationTests(unittest.TestCase):
    def test_valid_authorisation_opens_the_path(self) -> None:
        auth = LockedTestAuthorisation("a" * 64, "H2 or H3", "H2", "b" * 64)
        self.assertEqual(
            open_locked_test("/x/locked_test/c.csv", auth),
            Path("/x/locked_test/c.csv"),
        )

    def test_no_passing_hypothesis_is_refused(self) -> None:
        auth = LockedTestAuthorisation("a" * 64, "H2 or H3", "", "b" * 64)
        with self.assertRaises(LockedTestAccessError):
            open_locked_test("/x/locked_test/c.csv", auth)

    def test_more_than_one_evaluation_is_refused(self) -> None:
        auth = LockedTestAuthorisation("a" * 64, "r", "H2", "b" * 64, 2)
        with self.assertRaises(LockedTestAccessError):
            open_locked_test("/x/locked_test/c.csv", auth)

    def test_a_bare_string_cannot_stand_in_for_authorisation(self) -> None:
        with self.assertRaises(TypeError):
            open_locked_test("/x/locked_test/c.csv", "I promise it passed")

    def test_missing_protocol_hash_is_refused(self) -> None:
        auth = LockedTestAuthorisation("", "r", "H2", "b" * 64)
        with self.assertRaises(LockedTestAccessError):
            open_locked_test("/x/locked_test/c.csv", auth)


class AdvancementRuleTests(unittest.TestCase):
    def _setup(self, directory: Path, h2: bool, h3: bool) -> object:
        protocol = {
            "advancement_rule": {
                "advance_to_locked_test_only_if": "H2 passes OR H3 passes",
                "maximum_locked_test_evaluations": 1,
            }
        }
        protocol_path = write_json_atomic(directory / "protocol.json", protocol)
        part5 = write_json_atomic(
            directory / "part5.json", {"results": {"h2_verdict": {"passed": h2}}}
        )
        part6 = write_json_atomic(
            directory / "part6.json", {"results": {"h3_verdict": {"passed": h3}}}
        )

        from cxr_thesis.objective3_v2.io_utils import sha256_file

        class Args:
            pass

        args = Args()
        args.protocol = protocol_path
        args.expected_protocol_sha256 = sha256_file(protocol_path)
        args.part5_results = part5
        args.part6_results = part6
        return args

    def test_neither_hypothesis_keeps_the_test_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._setup(Path(directory), h2=False, h3=False)
            with self.assertRaises(LockedTestAccessError) as caught:
                final.check_advancement(args)
            message = str(caught.exception)
            self.assertIn("ADVANCEMENT RULE NOT MET", message)
            self.assertIn("negative result", message)

    def test_h2_alone_authorises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._setup(Path(directory), h2=True, h3=False)
            self.assertEqual(final.check_advancement(args).hypothesis_passed, "H2")

    def test_h3_alone_authorises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._setup(Path(directory), h2=False, h3=True)
            self.assertEqual(final.check_advancement(args).hypothesis_passed, "H3")

    def test_both_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._setup(Path(directory), h2=True, h3=True)
            self.assertEqual(final.check_advancement(args).hypothesis_passed, "H2+H3")

    def test_no_evidence_at_all_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._setup(Path(directory), h2=True, h3=True)
            args.part5_results = None
            args.part6_results = None
            with self.assertRaises(RuntimeError):
                final.check_advancement(args)

    def test_a_tampered_protocol_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._setup(Path(directory), h2=True, h3=False)
            args.expected_protocol_sha256 = "0" * 64
            with self.assertRaises(ValueError):
                final.check_advancement(args)


class SecondEvaluationTests(unittest.TestCase):
    def test_an_existing_lock_blocks_a_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(
                root / "final_lock.json",
                {"evaluated_at_utc": "2026-09-01T00:00:00+00:00",
                 "summary_sha256": "c" * 64},
            )
            with self.assertRaises(LockedTestAccessError) as caught:
                final.assert_never_evaluated(root)
            self.assertIn("already been evaluated", str(caught.exception))

    def test_a_clean_directory_permits_the_first_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            final.assert_never_evaluated(Path(directory))


@unittest.skipUnless(HAS_PENNYLANE, "pennylane is not installed")
class NoiseTests(unittest.TestCase):
    def _inputs(self, cases: int = 8, supernodes: int = 4, layers: int = 2):
        generator = np.random.default_rng(0)
        angles = generator.uniform(-np.pi, np.pi, size=(cases, supernodes))
        adjacency = np.zeros((cases, supernodes, supernodes))
        rotations = generator.uniform(-np.pi, np.pi, size=(layers, supernodes, 3))
        pairs = supernodes * (supernodes - 1) // 2
        couplings = generator.uniform(-np.pi, np.pi, size=(layers, pairs))
        return angles, adjacency, rotations, couplings

    def test_zero_noise_reproduces_the_ideal_circuit(self) -> None:
        angles, adjacency, rotations, couplings = self._inputs()
        ideal = noise.circuit_expectations(angles, adjacency, rotations, couplings)
        same = noise.circuit_expectations(
            angles, adjacency, rotations, couplings, depolarising=0.0
        )
        self.assertTrue(np.allclose(ideal, same, atol=1e-9))

    def test_depolarising_noise_shrinks_expectations(self) -> None:
        angles, adjacency, rotations, couplings = self._inputs()
        ideal = noise.circuit_expectations(angles, adjacency, rotations, couplings)
        noisy = noise.circuit_expectations(
            angles, adjacency, rotations, couplings, depolarising=0.2
        )
        # a depolarising channel pulls every expectation toward zero
        self.assertLess(
            float(np.mean(np.abs(noisy))), float(np.mean(np.abs(ideal)))
        )

    def test_expectations_stay_physical(self) -> None:
        angles, adjacency, rotations, couplings = self._inputs()
        for probability in (0.0, 0.01, 0.1):
            values = noise.circuit_expectations(
                angles, adjacency, rotations, couplings, depolarising=probability
            )
            self.assertLessEqual(float(np.max(np.abs(values))), 1.0 + 1e-9)


class ResourceTests(unittest.TestCase):
    class Args:
        supernodes = 4
        layers = 2
        validation_cases = 5000
        precision = 0.01

    def test_shots_follow_the_precision_target(self) -> None:
        table = noise.resource_table(self.Args(), {"rows": []})
        # standard error <= 1/sqrt(shots), so 0.01 precision needs 1e4 shots
        self.assertEqual(table["shots_for_target_precision"], 10_000)

    def test_gate_counts_match_the_circuit(self) -> None:
        table = noise.resource_table(self.Args(), {"rows": []})
        self.assertEqual(table["qubits"], 4)
        self.assertEqual(table["two_qubit_gates"], 2 * 6)
        self.assertEqual(table["single_qubit_gates"], 2 * (4 + 4))

    def test_assumptions_are_recorded_not_hidden(self) -> None:
        table = noise.resource_table(self.Args(), {"rows": []})
        for key in (
            "single_qubit_gate_seconds",
            "two_qubit_gate_seconds",
            "readout_seconds",
            "source",
        ):
            self.assertIn(key, table["assumptions"])
        self.assertIn("estimated", table["caveat"])


if __name__ == "__main__":
    unittest.main()
