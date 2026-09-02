#!/usr/bin/env python3
"""Part 4 Job 0: the sizing pilot that the v2.0 preregistration depends on.

Runs the unchanged v1.1 pipeline for a set of pilot seeds, measures the per-seed
delta standard deviation, and writes the results JSON that
``lock_objective3_v2_protocol.py`` consumes to choose the equivalence margin and
seed count.

Nothing here is evidence for any hypothesis. These seeds size the study and are
then discarded; the study re-runs on a disjoint seed list. That is what keeps the
pilot from biasing the test it sizes.

It also reports wall-clock per run, separately for the quantum and classical
arms, so ``--max-seeds`` can be chosen from measurement rather than optimism.
Completed runs are detected and skipped, so a Kaggle restart resumes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective3_v2 import STUDY, VERSION
from cxr_thesis.objective3_v2.guards import assert_no_locked_test, require_existing
from cxr_thesis.objective3_v2.io_utils import write_results
from cxr_thesis.objective3_v2.stats import (
    min_detectable_effect,
    paired_ttest,
    required_pairs,
)

VARIANTS = ("classical_matched", "quantum")
# train_objective3_head.py defaults --architecture to v1_concat (the v1.0 design).
# The pilot must size the v1.1 pipeline, so this is passed explicitly on every run.
ARCHITECTURE = "v1_1_reupload_gated"
PILOT_SEED_BASE = 900_042
CANDIDATE_MARGINS = (0.005, 0.010)
SD_UPPER_BOUND_CONFIDENCE = 0.80
# The pilot calls the plain trainer, not the private-recovery wrapper. Its seeds
# are discarded before the study, each run takes about two minutes, and the
# wrapper commits a checkpoint to Hugging Face every epoch - which exhausts the
# 128-commits-per-hour limit long before ten seeds finish.
TRAINER = "scripts/train_objective3_head.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--hf-repo",
        help="Unused by the pilot; accepted so existing commands keep working",
    )
    parser.add_argument(
        "--hf-base-path",
        default=f"objective3_v2/sizing_pilot/{ARCHITECTURE}/v2.0.0",
        help=(
            "Private recovery prefix. The architecture is part of the path so a "
            "run of one architecture can never recover another's checkpoints."
        ),
    )
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-gat-sha256", required=True)
    parser.add_argument("--pilot-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training and aggregate whatever runs already completed",
    )
    return parser.parse_args()


def pilot_seeds(count: int) -> list[int]:
    """Pilot seeds live far from the study seeds (42, 1042, ...) by construction."""

    return [PILOT_SEED_BASE + index for index in range(count)]


def read_completed(output: Path) -> dict[str, object] | None:
    """Return a finished run's summary, or None if it has not completed."""

    summary_path = output / "validation_summary.json"
    if not summary_path.is_file():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def check_summary(summary: dict[str, object], variant: str, seed: int) -> None:
    """Refuse a run that drifted from the v1.1 architecture or touched the test set."""

    checks = {
        "architecture": summary.get("architecture_version") == ARCHITECTURE,
        "variant": summary.get("variant") == variant,
        "seed": summary.get("seed") == seed,
        "test_cases": summary.get("test_cases_accessed") == 0,
        "test_evaluated": summary.get("test_evaluated") is False,
        "bottleneck_parameters": summary.get("bottleneck_parameters") == 36,
        "total_parameters": summary.get("total_trainable_parameters") == 3253,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        found = summary.get("architecture_version")
        lines = [f"Run {variant}/seed{seed} failed checks: {failed}"]
        if found is not None and found != ARCHITECTURE:
            lines += [
                f"  This run used architecture {found!r}, not {ARCHITECTURE!r}.",
                "  It is stale output from a run that omitted --architecture.",
                "  Delete that run's directory, then re-run the pilot:",
                f"    rm -rf <output-root>/{variant}/seed{seed}",
            ]
        raise RuntimeError("\n".join(lines))


def run_one(args: argparse.Namespace, variant: str, seed: int) -> dict[str, object]:
    """Train one variant at one seed, or reuse it if already complete."""

    output = args.output_root / variant / f"seed{seed}"

    existing = read_completed(output)
    if existing is not None:
        check_summary(existing, variant, seed)
        print(f"--- REUSING {variant} seed {seed} (already complete) ---", flush=True)
        elapsed = None
    else:
        if args.aggregate_only:
            raise FileNotFoundError(
                f"--aggregate-only was set but {variant}/seed{seed} has no summary"
            )
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / TRAINER),
            "--variant", variant,
            "--architecture", ARCHITECTURE,
            "--train-manifest", str(args.train_manifest),
            "--val-manifest", str(args.val_manifest),
            "--embedding-root", str(args.embedding_root),
            "--output-dir", str(output),
            "--expected-train-sha256", args.expected_train_sha256,
            "--expected-val-sha256", args.expected_val_sha256,
            "--expected-gat-sha256", args.expected_gat_sha256,
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--dropout", str(args.dropout),
            "--seed", str(seed),
        ]
        print(f"\n--- STARTING {variant} seed {seed} ---", flush=True)
        started = time.perf_counter()
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        elapsed = time.perf_counter() - started
        print(f"--- FINISHED {variant} seed {seed} in {elapsed:.0f}s ---", flush=True)
        existing = read_completed(output)
        if existing is None:
            raise RuntimeError(f"{variant}/seed{seed} produced no summary")
        check_summary(existing, variant, seed)

    macro = existing["validation_metrics"]["macro"]
    return {
        "variant": variant,
        "seed": int(seed),
        "best_epoch": existing["best_epoch"],
        "validation_macro_auroc": float(macro["auroc"]),
        "validation_macro_auprc": float(macro["auprc"]),
        "validation_macro_f1": float(macro["f1"]),
        "wall_clock_seconds": elapsed,
        "test_evaluated": False,
    }


def summarise(runs: list[dict[str, object]], seeds: list[int]) -> dict[str, object]:
    """Compute the delta SD and the seed counts each candidate margin would need."""

    from scipy import stats as scipy_stats

    indexed = {(run["variant"], run["seed"]): run for run in runs}
    classical = np.array(
        [indexed[("classical_matched", s)]["validation_macro_auroc"] for s in seeds]
    )
    quantum = np.array(
        [indexed[("quantum", s)]["validation_macro_auroc"] for s in seeds]
    )
    timings = {}
    for variant in VARIANTS:
        measured = [
            run["wall_clock_seconds"]
            for run in runs
            if run["variant"] == variant and run["wall_clock_seconds"] is not None
        ]
        timings[variant] = {
            "runs_timed": len(measured),
            "median_seconds": float(np.median(measured)) if measured else None,
            "max_seconds": float(np.max(measured)) if measured else None,
        }

    if len(seeds) < 3:
        # A one- or two-seed run is a timing probe, not a sizing measurement.
        # Report the clock and say plainly that no standard deviation exists yet.
        return {
            "pilot_seeds": [int(s) for s in seeds],
            "pilot_seed_count": len(seeds),
            "timing_probe_only": True,
            "delta_standard_deviation": None,
            "wall_clock": timings,
            "note": (
                "Fewer than three seeds: this run measured wall-clock only. "
                "Re-run with --pilot-seeds 10 to obtain the standard deviation "
                "that lock_objective3_v2_protocol.py requires."
            ),
            "quantum_minus_classical_by_seed": (quantum - classical).tolist(),
            "test_evaluated": False,
            "test_manifest_opened": False,
            "test_labels_accessed": False,
        }

    paired = paired_ttest(quantum, classical)
    deviation = paired.standard_deviation

    degrees = len(seeds) - 1
    critical = scipy_stats.chi2.ppf(1.0 - SD_UPPER_BOUND_CONFIDENCE, degrees)
    sizing_deviation = float(deviation * np.sqrt(degrees / critical))

    sizing = []
    for margin in CANDIDATE_MARGINS:
        sizing.append(
            {
                "margin": float(margin),
                "seeds_required_point_estimate": required_pairs(margin, deviation),
                "seeds_required_upper_bound": required_pairs(margin, sizing_deviation),
            }
        )

    return {
        "pilot_seeds": [int(s) for s in seeds],
        "pilot_seed_count": len(seeds),
        "classical_mean_validation_macro_auroc": float(classical.mean()),
        "quantum_mean_validation_macro_auroc": float(quantum.mean()),
        "quantum_minus_classical_by_seed": (quantum - classical).tolist(),
        "mean_quantum_minus_classical": paired.mean_difference,
        # The field lock_objective3_v2_protocol.py reads:
        "delta_standard_deviation": deviation,
        "sizing_sd_upper_confidence_bound": sizing_deviation,
        "sd_upper_bound_confidence": SD_UPPER_BOUND_CONFIDENCE,
        "minimum_detectable_effect_at_pilot_n": min_detectable_effect(
            deviation, len(seeds)
        ),
        "sizing_options": sizing,
        "wall_clock": timings,
        "purpose": "study sizing only; not evidence for any hypothesis",
        "pilot_seeds_discarded_before_study": True,
        "test_evaluated": False,
        "test_manifest_opened": False,
        "test_labels_accessed": False,
    }


def main() -> None:
    args = parse_args()
    require_existing(
        [args.train_manifest, args.val_manifest, args.embedding_root]
    )
    output_root = assert_no_locked_test(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = pilot_seeds(args.pilot_seeds)
    study_seeds = {42 + 1000 * index for index in range(200)}
    if set(seeds) & study_seeds:
        raise RuntimeError("Pilot seeds must be disjoint from the study seed list")

    runs = [run_one(args, variant, seed) for seed in seeds for variant in VARIANTS]
    results = summarise(runs, seeds)
    results["runs"] = runs

    path, digest = write_results(
        output_root / "results.json",
        study=STUDY,
        part="part4_job0_sizing_pilot",
        config={
            "version": VERSION,
            "variants": list(VARIANTS),
            "architecture": f"{ARCHITECTURE} (unchanged from v1.1)",
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "train_cohort_sha256": args.expected_train_sha256,
            "val_cohort_sha256": args.expected_val_sha256,
            "frozen_gat_sha256": args.expected_gat_sha256,
        },
        results=results,
        seed=seeds[0],
        locked_test_accessed=False,
    )

    print("\n" + "=" * 70)
    print("SIZING PILOT COMPLETE")
    print("=" * 70)
    print(f"Pilot seeds                : {len(seeds)} ({seeds[0]}-{seeds[-1]})")

    if results.get("timing_probe_only"):
        print("Timing probe only: fewer than three seeds, so no standard deviation.")
        for variant, timing in results["wall_clock"].items():
            median = timing["median_seconds"]
            shown = "n/a" if median is None else f"{median:.0f}s"
            print(f"  {variant:<18} {shown}")
        pair = sum(
            timing["median_seconds"] or 0.0
            for timing in results["wall_clock"].values()
        )
        if pair > 0:
            print(f"  {'per seed pair':<18} {pair:.0f}s")
            for count in (10, 20, 30, 50):
                print(f"    {count:>3} seeds -> {count * pair / 3600.0:5.2f} h")
        print("")
        print(f"Results: {path}")
        print(f"Results SHA-256: {digest}")
        print("")
        print("Re-run with --pilot-seeds 10 to size the study.")
        return

    print(f"Classical mean AUROC       : {results['classical_mean_validation_macro_auroc']:.4f}")
    print(f"Quantum mean AUROC         : {results['quantum_mean_validation_macro_auroc']:.4f}")
    print(f"Mean delta                 : {results['mean_quantum_minus_classical']:+.4f}")
    print(f"Per-seed delta SD          : {results['delta_standard_deviation']:.6f}")
    print(f"Sizing SD (80% upper bound): {results['sizing_sd_upper_confidence_bound']:.6f}")
    print("\n--- SEEDS THE STUDY WOULD NEED ---")
    for option in results["sizing_options"]:
        print(
            f"  margin +/-{option['margin']:<6} "
            f"point estimate {option['seeds_required_point_estimate']:>4}   "
            f"upper bound {option['seeds_required_upper_bound']:>4}  <-- use this"
        )
    print("\n--- WALL CLOCK PER RUN ---")
    for variant, timing in results["wall_clock"].items():
        median = timing["median_seconds"]
        print(
            f"  {variant:<18} median "
            f"{'n/a (all reused)' if median is None else f'{median:.0f}s'}"
        )
    quantum_median = results["wall_clock"]["quantum"]["median_seconds"]
    classical_median = results["wall_clock"]["classical_matched"]["median_seconds"]
    if quantum_median and classical_median:
        print("\n--- BUDGET: total hours for the full study, both arms ---")
        for option in results["sizing_options"]:
            count = option["seeds_required_upper_bound"]
            hours = count * (quantum_median + classical_median) / 3600.0
            print(f"  margin +/-{option['margin']:<6} {count:>4} seeds -> {hours:6.1f} h")
        print("\n  Pick --max-seeds so this fits your remaining Kaggle time.")
    print(f"\nResults: {path}")
    print(f"Results SHA-256: {digest}")
    print("Test evaluated: False | Locked test accessed: False")
    print("\nNext: pass this file to scripts/lock_objective3_v2_protocol.py "
          "as --pilot-results")


if __name__ == "__main__":
    main()
