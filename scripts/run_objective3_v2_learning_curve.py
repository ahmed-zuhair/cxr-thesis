#!/usr/bin/env python3
"""Part 5: does a quantum advantage appear when data is scarce?

Tests H2 of the v2.0 protocol. Macro AUROC at full training size is where
classical models are strongest and where v1.0 and v1.1 both found nothing. The
plausible remaining place for an advantage is the small-sample regime, which is
also the clinically interesting one: rare pathologies have few labelled cases.

Three arms, all parameter-matched at 36 bottleneck angles:

* ``quantum``           the v1.1 re-uploading circuit
* ``classical_matched`` its classical control
* ``quantum_random``    the same circuit, angles frozen at random initialisation

The third arm is what makes a positive result interpretable. If quantum beats
classical but not quantum_random, the benefit came from the extra trained
parameters rather than from the quantum feature map.

Training subsets are prefixes of the frozen training cohort, taken through the
trainer's own --limit-train flag. This is deliberate rather than convenient: the
GAT embeddings are keyed to the full frozen manifest and its recorded hash, so
substituting a re-sampled subset manifest breaks the embedding recovery index,
as it must. Prefixes are nested by construction, which is the property the
curve needs, and the cohort order was itself fixed by a seeded selection, so a
prefix is not correlated with any label.

Validation is held fixed at full size throughout; only training size varies.
The locked test cohort is never opened.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective3_v2 import STUDY, VERSION
from cxr_thesis.objective3_v2.guards import assert_no_locked_test, require_existing
from cxr_thesis.objective3_v2.io_utils import (
    ShardLedger,
    sha256_file,
    verify_sha256,
    write_results,
)
from cxr_thesis.objective3_v2.seeds import protocol_seeds, seed_everything
from cxr_thesis.objective3_v2.stats import bootstrap_ci, paired_wilcoxon

PART = "part5_learning_curve"
ARCHITECTURE = "v1_1_reupload_gated"
TRAINER = "scripts/train_objective3_head.py"
VARIANTS = ("quantum", "classical_matched", "quantum_random")
DEFAULT_SIZES = (100, 250, 500, 1000, 2500, 5000, 10000, 30000)
SMALL_DATA_LIMIT = 1000
H2_THRESHOLD = 0.005
H2_REQUIRED_WINS = 7
DEFAULT_SEEDS = 10
SUBSAMPLE_SEED = 20250831


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-gat-sha256", required=True)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--n-train",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Training sizes to run; a subset lets you split across sessions",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild the analysis from completed runs without training",
    )
    parser.add_argument("--figures", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny synthetic run, no data required, under 60 seconds",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# cohort construction
# --------------------------------------------------------------------------


def prefix_audit(
    train_manifest: Path,
    val_manifest: Path,
    sizes: list[int],
) -> dict[str, Any]:
    """Verify the cohort supports the requested prefixes, and is disjoint.

    Nesting needs no check: a prefix of a fixed order is contained in every
    longer prefix by construction. What does need checking is that the cohorts
    are patient-disjoint and large enough, and that is asserted here rather than
    assumed.
    """

    train = pd.read_csv(train_manifest)
    validation = pd.read_csv(val_manifest)
    for frame, name in ((train, "train"), (validation, "validation")):
        for column in ("image_id", "patient_id"):
            if column not in frame.columns:
                raise ValueError(f"{name} manifest has no {column!r} column")

    overlap = set(train["patient_id"]) & set(validation["patient_id"])
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} patients appear in both training and validation"
        )
    for size in sizes:
        if size > len(train):
            raise ValueError(f"n_train={size} exceeds the {len(train)} training rows")

    return {
        "training_rows_available": int(len(train)),
        "validation_rows": int(len(validation)),
        "validation_fixed_at_full_size": True,
        "patient_overlap_train_validation": 0,
        "sizes": sorted(sizes),
        "subset_mechanism": "trainer --limit-train prefix of the frozen cohort",
        "nested_by_construction": True,
        "patients_per_size": {
            str(size): int(train.iloc[:size]["patient_id"].nunique())
            for size in sorted(sizes)
        },
    }


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def read_summary(output: Path) -> dict[str, Any] | None:
    path = output / "validation_summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def check_summary(summary: dict[str, Any], variant: str, seed: int) -> None:
    """Refuse a run that drifted from the v1.1 design or touched the test set."""

    checks = {
        "architecture": summary.get("architecture_version") == ARCHITECTURE,
        "variant": summary.get("variant") == variant,
        "seed": summary.get("seed") == seed,
        "test_cases": summary.get("test_cases_accessed") == 0,
        "test_evaluated": summary.get("test_evaluated") is False,
        "bottleneck_parameters": summary.get("bottleneck_parameters") == 36,
        "total_parameters": summary.get("total_trainable_parameters") == 3253,
    }
    if variant == "quantum_random" and summary.get("circuit_frozen") is not True:
        checks["circuit_frozen"] = False
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Run {variant}/n{seed} failed checks: {failed}")


def clear_partial_run(output: Path, variant: str, seed: object) -> None:
    """Remove a run directory that exists but never produced a summary.

    A directory with no validation_summary.json is a crashed run, not a result:
    the trainer refuses to write into an existing directory, so leaving it in
    place blocks the retry forever. Only incomplete runs are removed, and the
    removal is announced, so a completed result can never be discarded silently.
    """

    if not output.exists():
        return
    print(
        f"--- CLEARING partial {variant} seed {seed} (no summary; crashed run) ---",
        flush=True,
    )
    shutil.rmtree(output)


def run_one(
    args: argparse.Namespace,
    variant: str,
    size: int,
    seed: int,
) -> dict[str, Any]:
    """Train one (variant, n_train, seed) cell, or reuse a completed one."""

    output = args.output_root / f"n{size}" / variant / f"seed{seed}"
    existing = read_summary(output)
    if existing is not None:
        check_summary(existing, variant, seed)
        elapsed = None
    else:
        if args.aggregate_only:
            raise FileNotFoundError(
                f"--aggregate-only set but n{size}/{variant}/seed{seed} is missing"
            )
        clear_partial_run(output, variant, seed)
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
            "--limit-train", str(size),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch-size", str(min(args.batch_size, max(8, size))),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--dropout", str(args.dropout),
            "--seed", str(seed),
        ]
        started = time.perf_counter()
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        elapsed = time.perf_counter() - started
        existing = read_summary(output)
        if existing is None:
            raise RuntimeError(f"n{size}/{variant}/seed{seed} produced no summary")
        check_summary(existing, variant, seed)

    macro = existing["validation_metrics"]["macro"]
    return {
        "n_train": int(size),
        "variant": variant,
        "seed": int(seed),
        "best_epoch": existing["best_epoch"],
        "validation_macro_auroc": float(macro["auroc"]),
        "validation_macro_auprc": float(macro["auprc"]),
        "wall_clock_seconds": elapsed,
        "limit_train": int(size),
        # The trainer marks any --limit-train run research_result=False, since a
        # single subsampled run is not a research claim. The learning curve as a
        # whole is the claim; each point is one measurement inside it.
        "single_run_is_research_result": False,
        "test_evaluated": False,
    }


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def curve_and_deltas(runs: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Per-size means with CIs, and the paired quantum-minus-control deltas."""

    indexed: dict[tuple[int, str, int], float] = {
        (row["n_train"], row["variant"], row["seed"]): row["validation_macro_auroc"]
        for row in runs
    }
    sizes = sorted({row["n_train"] for row in runs})
    variants = sorted({row["variant"] for row in runs})

    curve = []
    for size in sizes:
        for variant in variants:
            values = np.array(
                [
                    indexed[(size, variant, seed)]
                    for seed in seeds
                    if (size, variant, seed) in indexed
                ]
            )
            if values.size == 0:
                continue
            interval = bootstrap_ci(np.mean, values, resamples=10_000, seed=size)
            curve.append(
                {
                    "n_train": size,
                    "variant": variant,
                    "seeds": int(values.size),
                    "mean_macro_auroc": float(values.mean()),
                    "ci95_low": interval.confidence_interval_95[0],
                    "ci95_high": interval.confidence_interval_95[1],
                }
            )

    deltas = []
    for size in sizes:
        for control in ("classical_matched", "quantum_random"):
            paired_seeds = [
                seed
                for seed in seeds
                if (size, "quantum", seed) in indexed
                and (size, control, seed) in indexed
            ]
            if len(paired_seeds) < 3:
                continue
            quantum = np.array([indexed[(size, "quantum", s)] for s in paired_seeds])
            other = np.array([indexed[(size, control, s)] for s in paired_seeds])
            difference = quantum - other
            test = paired_wilcoxon(quantum, other)
            interval = bootstrap_ci(np.mean, difference, resamples=10_000, seed=size)
            deltas.append(
                {
                    "n_train": size,
                    "comparison": f"quantum_minus_{control}",
                    "seeds": len(paired_seeds),
                    "mean_delta": test.mean_difference,
                    "ci95_low": interval.confidence_interval_95[0],
                    "ci95_high": interval.confidence_interval_95[1],
                    "wilcoxon_p": test.p_value,
                    "wins": int((difference > 0).sum()),
                    "wins_above_threshold": int(
                        (difference >= H2_THRESHOLD).sum()
                    ),
                    "bootstrap_p_report": interval.p_value_report,
                }
            )
    return {"curve": curve, "deltas": deltas}


def h2_verdict(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide H2 exactly as the protocol words it, and show the evidence."""

    evidence = [
        row
        for row in deltas
        if row["comparison"] == "quantum_minus_classical_matched"
        and row["n_train"] <= SMALL_DATA_LIMIT
    ]
    passing = [
        row for row in evidence if row["wins_above_threshold"] >= H2_REQUIRED_WINS
    ]
    return {
        "hypothesis": (
            f"quantum beats the matched classical control by >= {H2_THRESHOLD} "
            f"macro AUROC in >= {H2_REQUIRED_WINS} of 10 seeds at some "
            f"n_train <= {SMALL_DATA_LIMIT}"
        ),
        "small_data_sizes_tested": [row["n_train"] for row in evidence],
        "passed": bool(passing),
        "passing_sizes": [row["n_train"] for row in passing],
        "evidence": evidence,
        "note": (
            "A pass here licenses advancing to the locked test only in "
            "combination with the quantum_random control: if the same margin "
            "appears against classical but not against the frozen circuit, the "
            "gain came from trained parameters, not from the quantum feature map."
        ),
    }


def write_figures(analysis: dict[str, Any], output: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for variant in sorted({row["variant"] for row in analysis["curve"]}):
        rows = sorted(
            (r for r in analysis["curve"] if r["variant"] == variant),
            key=lambda r: r["n_train"],
        )
        sizes = [r["n_train"] for r in rows]
        means = [r["mean_macro_auroc"] for r in rows]
        axis.plot(sizes, means, marker="o", label=variant)
        axis.fill_between(
            sizes,
            [r["ci95_low"] for r in rows],
            [r["ci95_high"] for r in rows],
            alpha=0.15,
        )
    axis.set_xscale("log")
    axis.set_xlabel("training cases")
    axis.set_ylabel("validation macro AUROC")
    axis.set_title("Learning curve, bootstrap 95% CI across seeds")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "learning_curve.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    hashes["learning_curve"] = sha256_file(path)

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for comparison in sorted({r["comparison"] for r in analysis["deltas"]}):
        rows = sorted(
            (r for r in analysis["deltas"] if r["comparison"] == comparison),
            key=lambda r: r["n_train"],
        )
        sizes = [r["n_train"] for r in rows]
        axis.plot(sizes, [r["mean_delta"] for r in rows], marker="o", label=comparison)
        axis.fill_between(
            sizes,
            [r["ci95_low"] for r in rows],
            [r["ci95_high"] for r in rows],
            alpha=0.15,
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.axhline(
        H2_THRESHOLD, color="tab:red", linestyle="--", linewidth=1.0,
        label=f"H2 threshold ({H2_THRESHOLD})",
    )
    axis.set_xscale("log")
    axis.set_xlabel("training cases")
    axis.set_ylabel("quantum minus control, macro AUROC")
    axis.set_title("Paired difference against training size")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "delta_curve.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    hashes["delta_curve"] = sha256_file(path)
    return hashes


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------


def smoke(output_root: Path) -> None:
    """Exercise cohort construction and the analysis on synthetic runs."""

    root = assert_no_locked_test(output_root)
    root.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(0)

    train = pd.DataFrame(
        {
            "image_id": [f"img{i:05d}" for i in range(400)],
            "patient_id": [f"p{i // 2:05d}" for i in range(400)],
        }
    )
    validation = pd.DataFrame(
        {
            "image_id": [f"vimg{i:05d}" for i in range(100)],
            "patient_id": [f"vp{i:05d}" for i in range(100)],
        }
    )
    train_path = root / "train.csv"
    val_path = root / "val.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(val_path, index=False)

    sizes = [50, 100, 200]
    audit = prefix_audit(train_path, val_path, sizes)
    assert audit["nested_by_construction"]

    seeds = protocol_seeds(5)
    runs = []
    for size in sizes:
        for variant in VARIANTS:
            base = 0.60 + 0.02 * np.log10(size)
            offset = {"quantum": 0.002, "classical_matched": 0.0, "quantum_random": -0.004}
            for seed in seeds:
                runs.append(
                    {
                        "n_train": size,
                        "variant": variant,
                        "seed": seed,
                        "best_epoch": 10,
                        "validation_macro_auroc": float(
                            base + offset[variant] + generator.normal(0, 0.003)
                        ),
                        "validation_macro_auprc": 0.088,
                        "wall_clock_seconds": 1.0,
                        "test_evaluated": False,
                    }
                )

    analysis = curve_and_deltas(runs, seeds)
    verdict = h2_verdict(analysis["deltas"])
    print(f"curve rows: {len(analysis['curve'])}, delta rows: {len(analysis['deltas'])}")
    print(f"H2 passed on synthetic data: {verdict['passed']}")
    print("Nested by construction:", audit["nested_by_construction"])
    print("Patient overlap:", audit["patient_overlap_train_validation"])
    print("Test evaluated: False | Locked test accessed: False")
    print("LEARNING CURVE SMOKE PASSED")


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke(args.output_root)
        return

    output = assert_no_locked_test(args.output_root)
    require_existing([args.train_manifest, args.val_manifest, args.embedding_root])
    verify_sha256(args.train_manifest, args.expected_train_sha256)
    verify_sha256(args.val_manifest, args.expected_val_sha256)
    seed_record = seed_everything(SUBSAMPLE_SEED)

    sizes = sorted(args.n_train)
    seeds = protocol_seeds(args.seeds)
    audit = prefix_audit(args.train_manifest, args.val_manifest, sizes)

    ledger = ShardLedger(output / "index.json", study=STUDY, part=PART)
    runs: list[dict[str, Any]] = []
    total = len(sizes) * len(args.variants) * len(seeds)
    done = 0
    for size in sizes:
        for variant in args.variants:
            for seed in seeds:
                runs.append(run_one(args, variant, size, seed))
                done += 1
                print(
                    f"[{done}/{total}] n={size} {variant} seed={seed} "
                    f"auroc={runs[-1]['validation_macro_auroc']:.4f}",
                    flush=True,
                )

    analysis = curve_and_deltas(runs, seeds)
    verdict = h2_verdict(analysis["deltas"])
    results: dict[str, Any] = {
        "cohort_audit": audit,
        "curve": analysis["curve"],
        "deltas": analysis["deltas"],
        "h2_verdict": verdict,
        "runs": runs,
        "labels_used": True,
        "test_evaluated": False,
    }
    artifact_hashes = write_figures(analysis, output) if args.figures else {}

    path, digest = write_results(
        output / "results.json",
        study=STUDY,
        part=PART,
        config={
            "version": VERSION,
            "architecture": ARCHITECTURE,
            "variants": list(args.variants),
            "n_train": sizes,
            "seeds": seeds,
            "subsample_seed": SUBSAMPLE_SEED,
            "h2_threshold": H2_THRESHOLD,
            "h2_required_wins": H2_REQUIRED_WINS,
            "small_data_limit": SMALL_DATA_LIMIT,
            "seeding": seed_record,
        },
        results=results,
        artifact_hashes=artifact_hashes,
        seed=seeds[0],
        locked_test_accessed=False,
    )

    print("")
    print("--- PAIRED DELTA AGAINST TRAINING SIZE ---")
    print(f"{'n_train':>8} {'comparison':>34} {'delta':>9} {'95% CI':>20} {'wins>=thr':>10}")
    for row in analysis["deltas"]:
        interval = f"[{row['ci95_low']:+.4f},{row['ci95_high']:+.4f}]"
        print(
            f"{row['n_train']:>8} {row['comparison']:>34} {row['mean_delta']:>+9.4f} "
            f"{interval:>20} {row['wins_above_threshold']:>7}/{row['seeds']}"
        )
    print("")
    print(f"H2 PASSED: {verdict['passed']}")
    if verdict["passed"]:
        print(f"  at n_train = {verdict['passing_sizes']}")
        print(f"  {verdict['note']}")
    print("")
    print(f"Results: {path}")
    print(f"Results SHA-256: {digest}")
    print("Test evaluated: False | Locked test accessed: False")


if __name__ == "__main__":
    main()
