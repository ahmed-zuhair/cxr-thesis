#!/usr/bin/env python3
"""Part 4 Job 2: analyze the powered Objective 3 v2.0 validation comparison.

This is a pure aggregate analysis. It verifies the frozen protocol, pairs the
quantum and parameter-matched classical validation results by protocol seed,
performs the preregistered tests, and optionally writes two hashed figures.

Smoke check (synthetic aggregate inputs, normally under 60 seconds):

    python scripts/run_objective3_v2_statistical_analysis.py --smoke --figures
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective3_v2 import STUDY, VERSION
from cxr_thesis.objective3_v2.guards import assert_no_locked_test, require_existing
from cxr_thesis.objective3_v2.io_utils import (
    ShardLedger,
    read_json,
    sha256_bytes,
    sha256_file,
    verify_sha256,
    write_json_atomic,
    write_results,
)
from cxr_thesis.objective3_v2.seeds import seed_everything
from cxr_thesis.objective3_v2.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    mde_curve,
    min_detectable_effect,
    paired_ttest,
    paired_wilcoxon,
    required_pairs,
    tost_equivalence,
)

PART = "part4_statistical_analysis"
VARIANTS = ("classical_matched", "quantum")
PRIMARY_LABELS = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Consolidation",
    "Pneumothorax",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Emphysema",
    "Edema",
    "Fibrosis",
]
DEFAULT_SMOKE_OUTPUT = (
    REPOSITORY_ROOT / "results" / "objective3_v2" / "statistical_analysis" / "smoke"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-results", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prior-30seed", type=Path)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a synthetic aggregate-only analysis in under 60 seconds",
    )
    return parser.parse_args()


def _require_full_args(args: argparse.Namespace) -> None:
    required = (
        "validation_results",
        "protocol",
        "expected_protocol_sha256",
        "output_dir",
    )
    missing = [name for name in required if getattr(args, name, None) in (None, "")]
    if missing:
        raise ValueError("Missing required arguments: " + ", ".join(missing))
    if args.bootstrap_resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")


def _report_number(value: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Non-finite numbers are not permitted in results")
    if number != 0.0 and abs(number) < 0.0001:
        return float(f"{number:.4e}")
    return round(number, 4)


def _round_tree(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return _report_number(float(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _round_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_tree(item) for item in value]
    return value


def _fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(canonical)


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    required = [args.validation_results, args.protocol]
    if args.prior_30seed is not None:
        required.append(args.prior_30seed)
    require_existing(required)
    protocol_hash = verify_sha256(args.protocol, args.expected_protocol_sha256)
    protocol = read_json(args.protocol)
    validation = read_json(args.validation_results)
    validation_hash = sha256_file(args.validation_results)
    if protocol.get("study") != STUDY or protocol.get("version") != VERSION:
        raise RuntimeError("Protocol study/version mismatch")
    if validation.get("study") != STUDY:
        raise RuntimeError("Powered-validation study mismatch")
    if validation.get("part") != "part4_powered_validation":
        raise RuntimeError("Input is not a Part 4 powered-validation results JSON")
    if validation.get("locked_test_accessed") is not False:
        raise RuntimeError("Powered-validation input is not test-blind")
    result = validation.get("results", {})
    required_false = (
        "patient_identifiers_included",
        "image_identifiers_included",
        "case_level_predictions_included",
        "raw_reports_included",
        "images_included",
        "test_evaluated",
    )
    if any(result.get(field) is not False for field in required_false):
        raise RuntimeError("Powered-validation public-output safeguards are incomplete")
    if result.get("test_cases_accessed") != 0:
        raise RuntimeError("Powered-validation input records accessed test cases")
    if validation.get("config", {}).get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Powered-validation results used a different protocol")
    return protocol, validation, protocol_hash, validation_hash


def _paired_runs(
    protocol: dict[str, Any], validation: dict[str, Any]
) -> tuple[list[int], dict[tuple[str, int], dict[str, Any]]]:
    seeds = [int(seed) for seed in protocol["design"]["seed_list"]]
    runs = validation["results"].get("runs", [])
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        key = (str(run.get("variant")), int(run.get("seed")))
        if key in indexed:
            raise RuntimeError(f"Duplicate powered-validation run: {key}")
        indexed[key] = run
    expected = {(variant, seed) for variant in VARIANTS for seed in seeds}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise RuntimeError(f"Powered-validation pairing mismatch; missing={missing}, extra={extra}")
    return seeds, indexed


def _macro_analysis(
    protocol: dict[str, Any],
    seeds: list[int],
    indexed: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    classical = np.asarray(
        [indexed[("classical_matched", seed)]["validation_macro_auroc"] for seed in seeds],
        dtype=float,
    )
    quantum = np.asarray(
        [indexed[("quantum", seed)]["validation_macro_auroc"] for seed in seeds],
        dtype=float,
    )
    paired_t = paired_ttest(quantum, classical)
    wilcoxon = paired_wilcoxon(quantum, classical)
    margin = float(protocol["design"]["equivalence_margin"])
    tost = tost_equivalence(quantum, classical, margin=margin)
    deviation = paired_t.standard_deviation
    if deviation <= 0:
        raise RuntimeError("Observed paired SD is zero; power curve is undefined")
    curve = mde_curve(deviation, range(3, 101), power=0.8)
    study_n = len(seeds)
    return _round_tree(
        {
            "seed_count": study_n,
            "seed_list": seeds,
            "classical_mean_macro_auroc": float(classical.mean()),
            "quantum_mean_macro_auroc": float(quantum.mean()),
            "quantum_minus_classical_by_seed": (quantum - classical).tolist(),
            "paired_ttest": paired_t.as_dict(),
            "paired_wilcoxon": wilcoxon.as_dict(),
            "tost_equivalence": tost.as_dict(),
            "headline_claim": tost.sentence(),
            "observed_paired_standard_deviation": deviation,
            "power": {
                "target_power": 0.8,
                "mde_at_n3": min_detectable_effect(deviation, 3, power=0.8),
                "mde_at_study_n": min_detectable_effect(
                    deviation, study_n, power=0.8
                ),
                "mde_at_n30": min_detectable_effect(deviation, 30, power=0.8),
                "required_pairs_for_protocol_margin": required_pairs(
                    margin, deviation, power=0.8
                ),
                "curve": curve,
            },
        }
    )


def _label_differences(
    label: str,
    seeds: list[int],
    indexed: dict[tuple[str, int], dict[str, Any]],
) -> np.ndarray:
    label_index = PRIMARY_LABELS.index(label)
    classical = np.asarray(
        [indexed[("classical_matched", seed)]["per_label"][label_index]["auroc"] for seed in seeds],
        dtype=float,
    )
    quantum = np.asarray(
        [indexed[("quantum", seed)]["per_label"][label_index]["auroc"] for seed in seeds],
        dtype=float,
    )
    for variant, rows in (("classical_matched", classical), ("quantum", quantum)):
        labels = [indexed[(variant, seed)]["per_label"][label_index]["label"] for seed in seeds]
        if labels != [label] * len(seeds):
            raise RuntimeError(f"Per-label ordering drifted for {label}")
        if not np.isfinite(rows).all():
            raise RuntimeError(f"Non-finite AUROC for {label}")
    return quantum - classical


def _compute_label_row(
    *,
    label: str,
    prevalence: float,
    seeds: list[int],
    indexed: dict[tuple[str, int], dict[str, Any]],
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    differences = _label_differences(label, seeds, indexed)
    bootstrap = bootstrap_ci(
        np.mean,
        differences,
        resamples=resamples,
        seed=bootstrap_seed,
    )
    label_index = PRIMARY_LABELS.index(label)
    classical = [
        indexed[("classical_matched", seed)]["per_label"][label_index]["auroc"]
        for seed in seeds
    ]
    quantum = [
        indexed[("quantum", seed)]["per_label"][label_index]["auroc"]
        for seed in seeds
    ]
    wilcoxon = paired_wilcoxon(quantum, classical)
    low, high = bootstrap.confidence_interval_95
    return _round_tree(
        {
            "label": label,
            "prevalence": prevalence,
            "mean_quantum_minus_classical_auroc": bootstrap.point,
            "bootstrap_confidence_interval_95": [low, high],
            "bootstrap_resamples": bootstrap.resamples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_p_value_report": bootstrap.p_value_report,
            "confidence_interval_excludes_zero": bool(low > 0 or high < 0),
            "raw_p_value": wilcoxon.p_value,
            "raw_test": "paired Wilcoxon signed-rank",
        }
    )


def _load_or_compute_shard(
    *,
    ledger: ShardLedger,
    shard_dir: Path,
    shard_key: str,
    run_fingerprint: str,
    compute: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], Path, bool]:
    path = shard_dir / f"{run_fingerprint[:12]}__{shard_key}.json"
    if ledger.is_complete(shard_key, path):
        payload = read_json(path)
        if payload.get("run_fingerprint") == run_fingerprint:
            return payload["result"], path, True
    result = _round_tree(compute())
    write_json_atomic(
        path,
        {
            "study": STUDY,
            "part": PART,
            "run_fingerprint": run_fingerprint,
            "aggregate_only": True,
            "result": result,
        },
    )
    ledger.mark_complete(
        shard_key,
        path,
        run_fingerprint=run_fingerprint,
        aggregate_only=True,
    )
    return result, path, False


def _prior_analysis(path: Path, margin: float) -> dict[str, Any]:
    checked = assert_no_locked_test(path)
    with np.load(checked, allow_pickle=False) as payload:
        required = {"vqc", "clf", "vqc_pc", "clf_pc"}
        if not required.issubset(payload.files):
            raise RuntimeError("Prior-study NPZ is missing required arrays")
        vqc = np.asarray(payload["vqc"], dtype=float)
        clf = np.asarray(payload["clf"], dtype=float)
        vqc_pc = np.asarray(payload["vqc_pc"], dtype=float)
        clf_pc = np.asarray(payload["clf_pc"], dtype=float)
    if vqc.shape != (30,) or clf.shape != (30,):
        raise RuntimeError("Prior macro arrays must each have shape (30,)")
    if vqc_pc.shape != (30, 14) or clf_pc.shape != (30, 14):
        raise RuntimeError("Prior per-class arrays must each have shape (30, 14)")
    if not all(np.isfinite(array).all() for array in (vqc, clf, vqc_pc, clf_pc)):
        raise RuntimeError("Prior-study arrays must be finite")
    per_label = []
    raw_p = []
    for index in range(14):
        paired_t = paired_ttest(vqc_pc[:, index], clf_pc[:, index])
        wilcoxon = paired_wilcoxon(vqc_pc[:, index], clf_pc[:, index])
        raw_p.append(wilcoxon.p_value)
        per_label.append(
            {
                "label": f"prior_label_{index + 1:02d}",
                "paired_ttest": paired_t.as_dict(),
                "paired_wilcoxon": wilcoxon.as_dict(),
            }
        )
    adjusted = benjamini_hochberg(raw_p)
    for row, q_value in zip(per_label, adjusted):
        row["adjusted_q_value"] = q_value
    tost = tost_equivalence(vqc, clf, margin=margin)
    return _round_tree(
        {
            "status": "SEPARATE EXPLORATORY STUDY",
            "differences_from_preregistered_study": {
                "labels": 14,
                "embedding_dimension": 1024,
                "splits_are_different": True,
                "evaluation_schedule": (
                    "its evaluation cohort was evaluated at every improved-validation epoch"
                ),
            },
            "macro_paired_ttest": paired_ttest(vqc, clf).as_dict(),
            "macro_paired_wilcoxon": paired_wilcoxon(vqc, clf).as_dict(),
            "macro_tost_at_current_protocol_margin": tost.as_dict(),
            "per_label": per_label,
            "prior_study_merged_into_preregistered_result": False,
        }
    )


def write_figures(results: dict[str, Any], output: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hashes: dict[str, str] = {}
    curve = results["preregistered_analysis"]["macro"]["power"]["curve"]
    study_n = results["preregistered_analysis"]["macro"]["seed_count"]
    margin = results["protocol_equivalence_margin"]
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.plot(
        [row["pairs"] for row in curve],
        [row["min_detectable_effect"] for row in curve],
        color="tab:blue",
    )
    for count, color in ((3, "tab:red"), (study_n, "tab:green"), (30, "tab:orange")):
        axis.axvline(count, linestyle="--", color=color, label=f"n={count}")
    axis.axhline(margin, linestyle=":", color="black", label="protocol margin")
    axis.set_xlabel("paired seeds")
    axis.set_ylabel("minimum detectable macro-AUROC difference")
    axis.set_title("Powered comparison: minimum detectable effect")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = output / "minimum_detectable_effect.png"
    figure.savefig(path, dpi=200, metadata={"Software": "cxr-thesis Objective 3 v2.0"})
    plt.close(figure)
    hashes[path.name] = sha256_file(path)

    rows = results["preregistered_analysis"]["per_label"]
    labels = [row["label"] for row in rows]
    points = np.asarray([row["mean_quantum_minus_classical_auroc"] for row in rows])
    intervals = np.asarray([row["bootstrap_confidence_interval_95"] for row in rows])
    lower = points - intervals[:, 0]
    upper = intervals[:, 1] - points
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(8.0, 6.5))
    axis.errorbar(points, positions, xerr=[lower, upper], fmt="o", capsize=3)
    axis.axvline(0.0, color="black", linestyle="--")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("quantum − classical AUROC (bootstrap 95% CI)")
    axis.set_title("Per-label powered validation, ordered by prevalence")
    axis.grid(alpha=0.25, axis="x")
    figure.tight_layout()
    path = output / "per_label_forest.png"
    figure.savefig(path, dpi=200, metadata={"Software": "cxr-thesis Objective 3 v2.0"})
    plt.close(figure)
    hashes[path.name] = sha256_file(path)
    return hashes


def execute(args: argparse.Namespace) -> tuple[Path, str]:
    _require_full_args(args)
    protocol, validation, protocol_hash, validation_hash = _load_inputs(args)
    output = assert_no_locked_test(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_record = seed_everything(args.seed)
    seeds, indexed = _paired_runs(protocol, validation)
    margin = float(protocol["design"]["equivalence_margin"])
    prevalence_rows = validation["results"].get("validation_prevalence", [])
    prevalence = {str(row["label"]): float(row["prevalence"]) for row in prevalence_rows}
    if set(prevalence) != set(PRIMARY_LABELS):
        raise RuntimeError("Validation prevalence does not cover the 12 primary labels")

    fingerprint_config = {
        "protocol_sha256": protocol_hash,
        "validation_results_sha256": validation_hash,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "prior_sha256": sha256_file(args.prior_30seed) if args.prior_30seed else None,
    }
    run_fingerprint = _fingerprint(fingerprint_config)
    shard_dir = output / "analysis_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    ledger = ShardLedger(shard_dir / "index.json", study=STUDY, part=PART)
    artifact_hashes: dict[str, str] = {}
    computed = 0
    resumed = 0

    macro, path, was_resumed = _load_or_compute_shard(
        ledger=ledger,
        shard_dir=shard_dir,
        shard_key="macro",
        run_fingerprint=run_fingerprint,
        compute=lambda: _macro_analysis(protocol, seeds, indexed),
    )
    resumed += int(was_resumed)
    computed += int(not was_resumed)
    artifact_hashes[f"analysis_shards/{path.name}"] = sha256_file(path)

    per_label = []
    for index, label in enumerate(PRIMARY_LABELS):
        row, path, was_resumed = _load_or_compute_shard(
            ledger=ledger,
            shard_dir=shard_dir,
            shard_key=f"label_{index:02d}_{label}",
            run_fingerprint=run_fingerprint,
            compute=lambda label=label, index=index: _compute_label_row(
                label=label,
                prevalence=prevalence[label],
                seeds=seeds,
                indexed=indexed,
                resamples=args.bootstrap_resamples,
                bootstrap_seed=args.seed + index,
            ),
        )
        per_label.append(row)
        resumed += int(was_resumed)
        computed += int(not was_resumed)
        artifact_hashes[f"analysis_shards/{path.name}"] = sha256_file(path)

    q_values = benjamini_hochberg([float(row["raw_p_value"]) for row in per_label])
    for row, q_value in zip(per_label, q_values):
        row["adjusted_q_value"] = _report_number(q_value)
        row["fdr_significant_at_0_05"] = bool(q_value < 0.05)
    per_label.sort(key=lambda row: (-float(row["prevalence"]), str(row["label"])))

    prior: dict[str, Any] | None = None
    if args.prior_30seed is not None:
        prior, path, was_resumed = _load_or_compute_shard(
            ledger=ledger,
            shard_dir=shard_dir,
            shard_key="prior_exploratory",
            run_fingerprint=run_fingerprint,
            compute=lambda: _prior_analysis(args.prior_30seed, margin),
        )
        resumed += int(was_resumed)
        computed += int(not was_resumed)
        artifact_hashes[f"analysis_shards/{path.name}"] = sha256_file(path)

    artifact_hashes["analysis_shards/index.json"] = sha256_file(ledger.index_path)
    results: dict[str, Any] = {
        "protocol_equivalence_margin": margin,
        "preregistered_analysis": {
            "macro": macro,
            "per_label": per_label,
            "per_label_order": "descending validation prevalence",
            "fdr_method": "Benjamini-Hochberg across 12 paired Wilcoxon tests",
        },
        "prior_evidence": prior,
        "prior_study_merged_into_preregistered_result": False,
        "analysis_shards_computed": computed,
        "analysis_shards_resumed": resumed,
        "aggregate_only": True,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "case_level_predictions_included": False,
        "raw_reports_included": False,
        "images_included": False,
        "test_cases_accessed": 0,
        "test_evaluated": False,
    }
    results = _round_tree(results)
    if args.figures:
        artifact_hashes.update(write_figures(results, output))

    path, digest = write_results(
        output / "results.json",
        study=STUDY,
        part=PART,
        seed=args.seed,
        config={
            "version": VERSION,
            "protocol_sha256": protocol_hash,
            "validation_results_sha256": validation_hash,
            "run_fingerprint": run_fingerprint,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed_base": args.seed,
            "seeding": seed_record,
            "figures_written": bool(args.figures),
            "prior_30seed_sha256": fingerprint_config["prior_sha256"],
        },
        results=results,
        artifact_hashes=dict(sorted(artifact_hashes.items())),
        locked_test_accessed=False,
    )
    print(results["preregistered_analysis"]["macro"]["headline_claim"])
    print(f"Statistical analysis results: {path}")
    print(f"Results SHA-256: {digest}")
    print(f"Analysis shards computed: {computed} | resumed: {resumed}")
    print("Test evaluated: False | Locked test accessed: False")
    return path, digest


def _smoke_validation(seeds: list[int]) -> dict[str, Any]:
    runs = []
    generator = np.random.default_rng(42)
    prevalence = []
    for index, label in enumerate(PRIMARY_LABELS):
        prevalence.append({"label": label, "prevalence": 0.4 - 0.02 * index})
    for seed_index, seed in enumerate(seeds):
        baseline = 0.64 + generator.normal(0.0, 0.002)
        for variant in VARIANTS:
            offset = 0.001 + 0.0004 * seed_index if variant == "quantum" else 0.0
            per_label = []
            for index, label in enumerate(PRIMARY_LABELS):
                per_label.append(
                    {
                        "label": label,
                        "auroc": 0.58 + 0.02 * index + offset,
                        "auprc": 0.5,
                        "f1": 0.5,
                        "sensitivity": 0.55,
                        "specificity": 0.75,
                        "threshold": 0.45,
                    }
                )
            runs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "validation_macro_auroc": baseline + offset,
                    "validation_macro_auprc": 0.55,
                    "validation_macro_f1": 0.5,
                    "per_label": per_label,
                    "test_evaluated": False,
                }
            )
    return {
        "study": STUDY,
        "part": "part4_powered_validation",
        "seed": seeds[0],
        "config": {},
        "results": {
            "runs": runs,
            "validation_prevalence": prevalence,
            "patient_identifiers_included": False,
            "image_identifiers_included": False,
            "case_level_predictions_included": False,
            "raw_reports_included": False,
            "images_included": False,
            "test_cases_accessed": 0,
            "test_evaluated": False,
        },
        "artifact_hashes": {},
        "timestamp": "synthetic",
        "locked_test_accessed": False,
    }


def smoke(output_dir: Path | None, figures: bool) -> tuple[Path, str]:
    with tempfile.TemporaryDirectory(prefix="objective3_v2_part4_job2_") as directory:
        root = Path(directory)
        seeds = [42, 1042, 2042, 3042, 4042]
        protocol = {
            "study": STUDY,
            "version": VERSION,
            "design": {
                "seed_list": seeds,
                "seeds_per_configuration": len(seeds),
                "equivalence_margin": 0.005,
            },
        }
        protocol_path = write_json_atomic(root / "protocol.json", protocol)
        protocol_hash = sha256_file(protocol_path)
        validation = _smoke_validation(seeds)
        validation["config"]["protocol_sha256"] = protocol_hash
        validation_path = write_json_atomic(root / "validation_results.json", validation)
        args = SimpleNamespace(
            validation_results=validation_path,
            protocol=protocol_path,
            expected_protocol_sha256=protocol_hash,
            output_dir=output_dir or DEFAULT_SMOKE_OUTPUT,
            prior_30seed=None,
            figures=figures,
            bootstrap_resamples=200,
            seed=42,
        )
        result = execute(args)
        print("STATISTICAL ANALYSIS SMOKE PASSED")
        return result


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke(args.output_dir, args.figures)
        return
    execute(args)


if __name__ == "__main__":
    main()
