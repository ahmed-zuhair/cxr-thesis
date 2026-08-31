from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cxr_thesis.objective3_v2.guards import (
    LockedTestAccessError,
    assert_no_locked_test,
    require_existing,
)
from cxr_thesis.objective3_v2.io_utils import (
    ShardLedger,
    hash_directory,
    sha256_file,
    verify_sha256,
    write_json_atomic,
    write_results,
)
from cxr_thesis.objective3_v2.seeds import protocol_seeds, seed_everything
from cxr_thesis.objective3_v2.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    min_detectable_effect,
    paired_power,
    paired_ttest,
    paired_wilcoxon,
    required_pairs,
    tost_equivalence,
)


class GuardTests(unittest.TestCase):
    def test_locked_test_paths_are_refused(self) -> None:
        for path in (
            "/kaggle/outputs/locked_test/cohort.csv",
            "data/TEST_MANIFEST.csv",
            r"C:\work\test_labels.npz",
            "results/official_test/summary.json",
        ):
            with self.assertRaises(LockedTestAccessError):
                assert_no_locked_test(path)

    def test_ordinary_paths_are_allowed(self) -> None:
        for path in ("private/train_cohort_private.csv", "private/shards/s000.npz"):
            self.assertEqual(assert_no_locked_test(path), Path(path))

    def test_missing_inputs_fail_loudly(self) -> None:
        with self.assertRaises(FileNotFoundError):
            require_existing(["definitely/not/here.csv"])


class EquivalenceTests(unittest.TestCase):
    def test_identical_samples_are_equivalent(self) -> None:
        generator = np.random.default_rng(0)
        values = generator.normal(0.7, 0.01, size=30)
        result = tost_equivalence(values, values.copy(), margin=0.005)
        self.assertTrue(result.equivalent)
        self.assertIn("equivalent", result.sentence())

    def test_separation_of_three_margins_is_not_equivalent(self) -> None:
        generator = np.random.default_rng(1)
        first = generator.normal(0.70, 0.002, size=30)
        second = first - 0.015
        result = tost_equivalence(first, second, margin=0.005)
        self.assertFalse(result.equivalent)
        self.assertIn("could NOT be established", result.sentence())

    def test_tiny_difference_with_tight_variance_is_equivalent(self) -> None:
        generator = np.random.default_rng(2)
        first = generator.normal(0.70, 0.0005, size=40)
        second = first - 0.0002
        result = tost_equivalence(first, second, margin=0.005)
        self.assertTrue(result.equivalent)
        self.assertEqual(result.p_tost, max(result.p_lower, result.p_upper))

    def test_margin_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            tost_equivalence([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], margin=0.0)


class PowerTests(unittest.TestCase):
    def test_power_increases_with_pairs(self) -> None:
        low = paired_power(0.005, 0.012, 10)
        high = paired_power(0.005, 0.012, 60)
        self.assertLess(low, high)
        self.assertLess(high, 1.0)

    def test_mde_shrinks_as_seeds_grow(self) -> None:
        coarse = min_detectable_effect(0.012, 3)
        fine = min_detectable_effect(0.012, 30)
        self.assertGreater(coarse, fine)

    def test_mde_achieves_the_requested_power(self) -> None:
        effect = min_detectable_effect(0.012, 20, power=0.8)
        self.assertAlmostEqual(paired_power(effect, 0.012, 20), 0.8, places=3)

    def test_required_pairs_matches_the_phase_one_estimate(self) -> None:
        # Phase-1 GAT row: per-seed delta SD 0.0121, target effect 0.005.
        pairs = required_pairs(0.005, 0.0121)
        self.assertGreater(pairs, 40)
        self.assertLess(pairs, 60)
        self.assertGreaterEqual(paired_power(0.005, 0.0121, pairs), 0.8)
        self.assertLess(paired_power(0.005, 0.0121, pairs - 1), 0.8)

    def test_three_seeds_cannot_detect_the_preregistered_threshold(self) -> None:
        # The v1.1 amendment tested a 0.005 threshold with three seeds.
        self.assertGreater(min_detectable_effect(0.0121, 3), 0.005 * 3)


class PairedTestTests(unittest.TestCase):
    def test_paired_ttest_recovers_a_known_shift(self) -> None:
        # A constant offset would make every difference identical and collapse
        # the interval to a point, so add independent noise to the second arm.
        generator = np.random.default_rng(3)
        first = generator.normal(0.70, 0.01, size=40)
        second = first - 0.02 + generator.normal(0.0, 0.004, size=40)
        result = paired_ttest(first, second)
        self.assertAlmostEqual(result.mean_difference, 0.02, places=2)
        self.assertLess(result.p_value, 1e-6)
        low, high = result.confidence_interval_95
        self.assertLess(low, result.mean_difference)
        self.assertGreater(high, result.mean_difference)
        self.assertLess(low, 0.02)
        self.assertGreater(high, 0.02)

    def test_wilcoxon_agrees_on_direction(self) -> None:
        generator = np.random.default_rng(4)
        first = generator.normal(0.70, 0.01, size=30)
        second = first - 0.02
        self.assertGreater(paired_wilcoxon(first, second).mean_difference, 0)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_ttest([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_too_few_pairs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_ttest([1.0, 2.0], [1.0, 2.0])


class BootstrapTests(unittest.TestCase):
    def test_interval_brackets_the_mean_and_is_reproducible(self) -> None:
        generator = np.random.default_rng(5)
        sample = generator.normal(0.5, 0.1, size=200)
        first = bootstrap_ci(np.mean, sample, resamples=2000, seed=7)
        second = bootstrap_ci(np.mean, sample, resamples=2000, seed=7)
        self.assertEqual(first.confidence_interval_95, second.confidence_interval_95)
        low, high = first.confidence_interval_95
        self.assertLess(low, sample.mean())
        self.assertGreater(high, sample.mean())

    def test_p_value_report_never_claims_zero(self) -> None:
        sample = np.linspace(0.0, 1.0, 50)
        self.assertEqual(
            bootstrap_ci(np.mean, sample, resamples=10_000).p_value_report,
            "p < 0.0001",
        )


class MultipleComparisonTests(unittest.TestCase):
    def test_adjusted_values_are_monotone_and_bounded(self) -> None:
        raw = [0.001, 0.008, 0.02, 0.04, 0.3, 0.9]
        adjusted = benjamini_hochberg(raw)
        self.assertEqual(len(adjusted), len(raw))
        self.assertTrue(all(a >= r for a, r in zip(adjusted, raw)))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertEqual(adjusted, sorted(adjusted))

    def test_single_value_is_unchanged(self) -> None:
        self.assertAlmostEqual(benjamini_hochberg([0.03])[0], 0.03)

    def test_out_of_range_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            benjamini_hochberg([0.5, 1.5])


class SeedTests(unittest.TestCase):
    def test_seed_list_is_deterministic(self) -> None:
        self.assertEqual(protocol_seeds(4), [42, 1042, 2042, 3042])
        self.assertEqual(protocol_seeds(20)[:3], protocol_seeds(3))

    def test_seeding_is_reproducible(self) -> None:
        seed_everything(42)
        first = np.random.rand(5)
        seed_everything(42)
        self.assertTrue(np.array_equal(first, np.random.rand(5)))


class IoTests(unittest.TestCase):
    def test_results_written_with_schema_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            written, digest = write_results(
                path,
                study="objective3_v2",
                part="part0_smoke",
                config={"seed": 42},
                results={"macro_auroc": 0.6543},
                seed=42,
            )
            payload = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(payload["study"], "objective3_v2")
            self.assertFalse(payload["locked_test_accessed"])
            self.assertIn("timestamp", payload)
            self.assertEqual(digest, sha256_file(written))
            sidecar = written.with_name(written.name + ".sha256")
            self.assertTrue(sidecar.is_file())
            self.assertIn(digest, sidecar.read_text(encoding="utf-8"))

    def test_verify_sha256_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_json_atomic(Path(directory) / "a.json", {"value": 1})
            digest = sha256_file(path)
            self.assertEqual(verify_sha256(path, digest), digest)
            write_json_atomic(path, {"value": 2})
            with self.assertRaises(ValueError):
                verify_sha256(path, digest)

    def test_ledger_resumes_only_intact_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "shard_000.json"
            write_json_atomic(shard, {"rows": 1000})
            ledger = ShardLedger(
                root / "index.json", study="objective3_v2", part="part5"
            )
            self.assertFalse(ledger.is_complete("shard_000", shard))
            ledger.mark_complete("shard_000", shard, start=0, stop=1000)
            self.assertTrue(ledger.is_complete("shard_000", shard))

            reopened = ShardLedger(
                root / "index.json", study="objective3_v2", part="part5"
            )
            self.assertEqual(reopened.completed, ["shard_000"])
            self.assertTrue(reopened.is_complete("shard_000", shard))

            # A shard rewritten by a killed session must not be trusted.
            write_json_atomic(shard, {"rows": 7})
            self.assertFalse(reopened.is_complete("shard_000", shard))

    def test_ledger_ignores_an_index_from_another_part(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "s.json"
            write_json_atomic(shard, {"rows": 1})
            ShardLedger(
                root / "index.json", study="objective3_v2", part="part5"
            ).mark_complete("s", shard)
            other = ShardLedger(
                root / "index.json", study="objective3_v2", part="part6"
            )
            self.assertEqual(other.completed, [])

    def test_hash_directory_lists_every_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(root / "a.json", {"a": 1})
            write_json_atomic(root / "nested" / "b.json", {"b": 2})
            hashes = hash_directory(root)
            self.assertEqual(set(hashes), {"a.json", "nested/b.json"})
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
