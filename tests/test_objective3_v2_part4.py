from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cxr_thesis.objective3_v2.io_utils import sha256_file, write_json_atomic

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPOSITORY_ROOT / "scripts" / filename
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


powered = load_script(
    "objective3_v2_powered_validation",
    "run_objective3_v2_powered_validation.py",
)
analysis = load_script(
    "objective3_v2_statistical_analysis",
    "run_objective3_v2_statistical_analysis.py",
)


class PoweredValidationTests(unittest.TestCase):
    def test_valid_summary_passes_all_drift_checks(self) -> None:
        summary = powered._fake_summary("quantum", 42, 0.001)
        powered.check_summary(summary, "quantum", 42)

    def test_architecture_drift_is_rejected(self) -> None:
        summary = powered._fake_summary("quantum", 42, 0.001)
        summary["architecture_version"] = "changed"
        with self.assertRaises(RuntimeError):
            powered.check_summary(summary, "quantum", 42)

    def test_protocol_rejects_overlapping_pilot_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            write_json_atomic(
                path,
                {
                    "study": "objective3_v2",
                    "version": "v2.0.0",
                    "training_started": False,
                    "design": {
                        "seed_list": [42, 1042, 2042],
                        "seeds_per_configuration": 3,
                    },
                    "sizing_pilot": {"pilot_seeds": [2042]},
                },
            )
            with self.assertRaises(RuntimeError):
                powered._load_protocol(path, sha256_file(path))

    def test_protocol_rejects_started_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            write_json_atomic(
                path,
                {
                    "study": "objective3_v2",
                    "version": "v2.0.0",
                    "training_started": True,
                    "design": {
                        "seed_list": [42, 1042, 2042],
                        "seeds_per_configuration": 3,
                    },
                    "sizing_pilot": {"pilot_seeds": [900_042]},
                },
            )
            with self.assertRaises(RuntimeError):
                powered._load_protocol(path, sha256_file(path))


class StatisticalAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seeds = [42, 1042, 2042, 3042, 4042]
        self.protocol = {
            "design": {
                "seed_list": self.seeds,
                "equivalence_margin": 0.005,
            }
        }
        validation = {"results": analysis._smoke_validation(self.seeds)["results"]}
        self.seeds, self.indexed = analysis._paired_runs(self.protocol, validation)

    def test_macro_analysis_uses_protocol_margin(self) -> None:
        result = analysis._macro_analysis(self.protocol, self.seeds, self.indexed)
        self.assertEqual(result["tost_equivalence"]["margin"], 0.005)
        self.assertEqual(result["seed_list"], self.seeds)
        self.assertEqual(
            result["headline_claim"],
            analysis.tost_equivalence(
                [
                    self.indexed[("quantum", seed)]["validation_macro_auroc"]
                    for seed in self.seeds
                ],
                [
                    self.indexed[("classical_matched", seed)][
                        "validation_macro_auroc"
                    ]
                    for seed in self.seeds
                ],
                margin=0.005,
            ).sentence(),
        )

    def test_per_label_bootstrap_is_aggregate_and_seeded(self) -> None:
        row = analysis._compute_label_row(
            label=analysis.PRIMARY_LABELS[0],
            prevalence=0.2,
            seeds=self.seeds,
            indexed=self.indexed,
            resamples=200,
            bootstrap_seed=42,
        )
        self.assertEqual(row["bootstrap_seed"], 42)
        self.assertEqual(row["bootstrap_resamples"], 200)
        self.assertNotIn("predictions", row)
        self.assertNotIn("identifiers", row)

    def test_prior_study_is_never_merged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.npz"
            generator = np.random.default_rng(7)
            clf = generator.normal(0.65, 0.01, size=30)
            vqc = clf + generator.normal(0.001, 0.003, size=30)
            clf_pc = generator.normal(0.65, 0.02, size=(30, 14))
            vqc_pc = clf_pc + generator.normal(0.001, 0.004, size=(30, 14))
            np.savez(path, vqc=vqc, clf=clf, vqc_pc=vqc_pc, clf_pc=clf_pc)
            result = analysis._prior_analysis(path, margin=0.005)
            self.assertFalse(result["prior_study_merged_into_preregistered_result"])
            self.assertEqual(result["status"], "SEPARATE EXPLORATORY STUDY")
            self.assertEqual(len(result["per_label"]), 14)


if __name__ == "__main__":
    unittest.main()
