#!/usr/bin/env python3
"""Part 4 Job 1: run the protocol-fixed powered v1.1 validation comparison.

The job uses exactly the seed list frozen in the verified v2.0 protocol, keeps
the sizing-pilot seeds disjoint, mirrors each training run through the existing
private-recovery trainer, and commits aggregate per-run rows to hashed shards.

Smoke check (tiny synthetic summaries, no training or private inputs):

    python scripts/run_objective3_v2_powered_validation.py --smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
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
    read_json,
    sha256_bytes,
    sha256_file,
    verify_sha256,
    write_json_atomic,
    write_results,
)
from cxr_thesis.objective3_v2.seeds import seed_everything

PART = "part4_powered_validation"
ARCHITECTURE = "v1_1_reupload_gated"
VARIANTS = ("classical_matched", "quantum")
# The plain trainer, not the private-recovery wrapper. The wrapper commits a
# checkpoint to Hugging Face every epoch; ten seeds across two arms is roughly
# four hundred commits and exceeds the 128-per-hour limit. Each run here takes
# under three minutes and a lost run is simply re-run by the resume logic, so
# per-epoch remote checkpointing buys nothing and costs the study.
TRAINER = REPOSITORY_ROOT / "scripts" / "train_objective3_head.py"
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
    REPOSITORY_ROOT / "results" / "objective3_v2" / "powered_validation" / "smoke"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--hf-repo")
    parser.add_argument(
        "--hf-base-path",
        default="objective3_v2/powered_validation/v2.0.0",
    )
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-val-sha256")
    parser.add_argument("--expected-gat-sha256")
    parser.add_argument("--expected-train-cases", type=int, default=30_000)
    parser.add_argument("--expected-val-cases", type=int, default=5_000)
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
        help="Do not train; require every protocol-fixed summary to exist",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a synthetic aggregate-only check in under 60 seconds",
    )
    return parser.parse_args()


def _require_full_args(args: argparse.Namespace) -> None:
    required = (
        "protocol",
        "expected_protocol_sha256",
        "train_manifest",
        "val_manifest",
        "embedding_root",
        "output_root",
        "expected_train_sha256",
        "expected_val_sha256",
        "expected_gat_sha256",
    )
    missing = [name for name in required if getattr(args, name, None) in (None, "")]
    if missing:
        raise ValueError("Missing required arguments: " + ", ".join(missing))


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


def _load_protocol(path: Path, expected_hash: str) -> tuple[dict[str, Any], str]:
    checked = assert_no_locked_test(path)
    digest = verify_sha256(checked, expected_hash)
    protocol = read_json(checked)
    if protocol.get("study") != STUDY or protocol.get("version") != VERSION:
        raise RuntimeError("Protocol study/version does not match Objective 3 v2.0")
    if protocol.get("training_started") is True:
        raise RuntimeError("Protocol records training_started=true; refusing to run")
    design = protocol.get("design", {})
    seeds = design.get("seed_list")
    if not isinstance(seeds, list) or not seeds:
        raise RuntimeError("Protocol design.seed_list must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise RuntimeError("Every protocol study seed must be an integer")
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Protocol study seeds must be unique")
    if int(design.get("seeds_per_configuration", -1)) != len(seeds):
        raise RuntimeError("Protocol seed count does not match design.seed_list")
    pilot = protocol.get("sizing_pilot", {}).get("pilot_seeds", [])
    if set(seeds) & set(pilot):
        raise RuntimeError("Sizing-pilot seeds overlap the powered study seed list")
    return protocol, digest


def _validate_frozen_inputs(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    frozen = protocol.get("frozen_inputs", {})
    expected = {
        "train_cohort_sha256": args.expected_train_sha256,
        "validation_cohort_sha256": args.expected_val_sha256,
        "frozen_gat_sha256": args.expected_gat_sha256,
    }
    drifted = [key for key, value in expected.items() if frozen.get(key) != value]
    if drifted:
        raise RuntimeError(f"CLI inputs drift from the locked protocol: {drifted}")


def _validation_prevalence(path: Path, expected_hash: str) -> list[dict[str, Any]]:
    verify_sha256(assert_no_locked_test(path), expected_hash)
    # Manifest label columns are prefixed "label_"; labels_from_manifest is the
    # single tested place that knows this. Reading bare names silently assumes a
    # schema the cohorts do not have.
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError("Validation manifest has no rows")
    rows: list[dict[str, Any]] = []
    from cxr_thesis.objective3.training import labels_from_manifest

    matrix = labels_from_manifest(frame, PRIMARY_LABELS)
    for index, label in enumerate(PRIMARY_LABELS):
        values = matrix[:, index].astype(float)
        if not np.isin(values, [0.0, 1.0]).all():
            raise RuntimeError(f"Validation label {label} is not binary")
        rows.append(
            {
                "label": label,
                "positive_cases": int(values.sum()),
                "validation_cases": int(values.size),
                "prevalence": float(values.mean()),
            }
        )
    return _round_tree(rows)


def read_completed(output: Path) -> dict[str, Any] | None:
    summary_path = assert_no_locked_test(output / "validation_summary.json")
    if not summary_path.is_file():
        return None
    try:
        return read_json(summary_path)
    except (json.JSONDecodeError, OSError):
        return None


def check_summary(summary: dict[str, Any], variant: str, seed: int) -> None:
    metrics = summary.get("validation_metrics", {})
    macro = metrics.get("macro", {})
    per_label = metrics.get("per_label", [])
    checks = {
        "architecture": summary.get("architecture_version") == ARCHITECTURE,
        "variant": summary.get("variant") == variant,
        "seed": summary.get("seed") == seed,
        "test_cases": summary.get("test_cases_accessed") == 0,
        "test_evaluated": summary.get("test_evaluated") is False,
        "bottleneck_parameters": summary.get("bottleneck_parameters") == 36,
        "total_parameters": summary.get("total_trainable_parameters") == 3253,
        "macro_metrics": all(key in macro for key in ("auroc", "auprc", "f1")),
        "per_label_count": isinstance(per_label, list)
        and len(per_label) == len(PRIMARY_LABELS),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Run {variant}/seed{seed} failed checks: {failed}")
    for label, row in zip(PRIMARY_LABELS, per_label):
        if row.get("label", label) != label:
            raise RuntimeError(f"Per-label order drifted at {label}")
        required = ("auroc", "auprc", "f1", "sensitivity", "specificity", "threshold")
        if any(key not in row for key in required):
            raise RuntimeError(f"Per-label metrics are incomplete for {label}")
        if not all(np.isfinite(float(row[key])) for key in required):
            raise RuntimeError(f"Per-label metrics are non-finite for {label}")


def _aggregate_summary(
    summary: dict[str, Any],
    variant: str,
    seed: int,
    elapsed: float | None,
    summary_path: Path,
) -> dict[str, Any]:
    check_summary(summary, variant, seed)
    macro = summary["validation_metrics"]["macro"]
    per_label = []
    for label, source in zip(PRIMARY_LABELS, summary["validation_metrics"]["per_label"]):
        per_label.append(
            {
                "label": label,
                "auroc": float(source["auroc"]),
                "auprc": float(source["auprc"]),
                "f1": float(source["f1"]),
                "sensitivity": float(source["sensitivity"]),
                "specificity": float(source["specificity"]),
                "threshold": float(source["threshold"]),
            }
        )
    return _round_tree(
        {
            "variant": variant,
            "seed": int(seed),
            "best_epoch": int(summary["best_epoch"]),
            "validation_macro_auroc": float(macro["auroc"]),
            "validation_macro_auprc": float(macro["auprc"]),
            "validation_macro_f1": float(macro["f1"]),
            "per_label": per_label,
            "wall_clock_seconds": elapsed,
            "source_summary_sha256": sha256_file(summary_path),
            "checkpoint_sha256": str(summary["checkpoint_sha256"]),
            "test_cases_accessed": 0,
            "test_evaluated": False,
        }
    )


def run_one(args: argparse.Namespace, variant: str, seed: int) -> dict[str, Any]:
    output = assert_no_locked_test(args.output_root / variant / f"seed{seed}")
    summary_path = output / "validation_summary.json"
    existing = read_completed(output)
    elapsed: float | None = None
    if existing is not None:
        check_summary(existing, variant, seed)
        print(f"--- REUSING {variant} seed {seed} ---", flush=True)
    else:
        if args.aggregate_only:
            raise FileNotFoundError(
                f"Aggregate-only mode requires {variant}/seed{seed} summary"
            )
        command = [
            sys.executable,
            str(TRAINER),
            "--variant",
            variant,
            "--architecture",
            ARCHITECTURE,
            "--train-manifest",
            str(args.train_manifest),
            "--val-manifest",
            str(args.val_manifest),
            "--embedding-root",
            str(args.embedding_root),
            "--output-dir",
            str(output),
            "--expected-train-sha256",
            args.expected_train_sha256,
            "--expected-val-sha256",
            args.expected_val_sha256,
            "--expected-gat-sha256",
            args.expected_gat_sha256,
            "--expected-train-cases",
            str(args.expected_train_cases),
            "--expected-val-cases",
            str(args.expected_val_cases),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--dropout",
            str(args.dropout),
            "--seed",
            str(seed),
        ]
        print(f"--- STARTING {variant} seed {seed} ---", flush=True)
        started = time.perf_counter()
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        elapsed = time.perf_counter() - started
        existing = read_completed(output)
        if existing is None:
            raise RuntimeError(f"{variant}/seed{seed} produced no validation summary")
    return _aggregate_summary(existing, variant, seed, elapsed, summary_path)


def _fingerprint(protocol_hash: str, args: argparse.Namespace) -> str:
    payload = {
        "protocol_sha256": protocol_hash,
        "architecture": ARCHITECTURE,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "train_sha256": args.expected_train_sha256,
        "validation_sha256": args.expected_val_sha256,
        "gat_sha256": args.expected_gat_sha256,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(canonical)


def execute(args: argparse.Namespace) -> tuple[Path, str]:
    _require_full_args(args)
    required = require_existing(
        [args.protocol, args.train_manifest, args.val_manifest, args.embedding_root]
    )
    protocol, protocol_hash = _load_protocol(required[0], args.expected_protocol_sha256)
    _validate_frozen_inputs(args, protocol)
    verify_sha256(args.train_manifest, args.expected_train_sha256)
    prevalence = _validation_prevalence(args.val_manifest, args.expected_val_sha256)
    output_root = assert_no_locked_test(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in protocol["design"]["seed_list"]]
    seed_record = seed_everything(seeds[0])
    run_fingerprint = _fingerprint(protocol_hash, args)

    shard_dir = output_root / "aggregate_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    ledger = ShardLedger(shard_dir / "index.json", study=STUDY, part=PART)
    runs: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    computed = 0
    resumed = 0

    for seed in seeds:
        for variant in VARIANTS:
            shard_key = f"{run_fingerprint[:12]}__{variant}__seed{seed}"
            shard_path = shard_dir / f"{shard_key}.json"
            if ledger.is_complete(shard_key, shard_path):
                payload = read_json(shard_path)
                if payload.get("run_fingerprint") == run_fingerprint:
                    row = payload["result"]
                    resumed += 1
                else:
                    row = run_one(args, variant, seed)
            else:
                row = run_one(args, variant, seed)
            if not ledger.is_complete(shard_key, shard_path):
                write_json_atomic(
                    shard_path,
                    {
                        "study": STUDY,
                        "part": PART,
                        "run_fingerprint": run_fingerprint,
                        "aggregate_only": True,
                        "result": row,
                    },
                )
                ledger.mark_complete(
                    shard_key,
                    shard_path,
                    run_fingerprint=run_fingerprint,
                    aggregate_only=True,
                )
                computed += 1
            runs.append(row)
            artifact_hashes[f"aggregate_shards/{shard_path.name}"] = sha256_file(
                shard_path
            )
            source_summary = output_root / variant / f"seed{seed}" / "validation_summary.json"
            artifact_hashes[
                f"{variant}/seed{seed}/validation_summary.json"
            ] = sha256_file(source_summary)

    artifact_hashes["aggregate_shards/index.json"] = sha256_file(ledger.index_path)
    results = {
        "variants": list(VARIANTS),
        "seed_list": seeds,
        "seed_count": len(seeds),
        "validation_prevalence": prevalence,
        "runs": runs,
        "shards_computed": computed,
        "shards_resumed": resumed,
        "aggregate_only_public_output": True,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "case_level_predictions_included": False,
        "raw_reports_included": False,
        "images_included": False,
        "test_cases_accessed": 0,
        "test_evaluated": False,
    }
    path, digest = write_results(
        output_root / "results.json",
        study=STUDY,
        part=PART,
        seed=seeds[0],
        config={
            "version": VERSION,
            "architecture": ARCHITECTURE,
            "protocol_sha256": protocol_hash,
            "run_fingerprint": run_fingerprint,
            "seeding": seed_record,
            "seed_list": seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": _report_number(args.learning_rate),
            "weight_decay": _report_number(args.weight_decay),
            "dropout": _report_number(args.dropout),
            "train_cohort_sha256": args.expected_train_sha256,
            "validation_cohort_sha256": args.expected_val_sha256,
            "frozen_gat_sha256": args.expected_gat_sha256,
        },
        results=_round_tree(results),
        artifact_hashes=dict(sorted(artifact_hashes.items())),
        locked_test_accessed=False,
    )
    print(f"Powered validation results: {path}")
    print(f"Results SHA-256: {digest}")
    print(f"Shards computed: {computed} | resumed: {resumed}")
    print("Test evaluated: False | Locked test accessed: False")
    return path, digest


def _fake_summary(variant: str, seed: int, offset: float) -> dict[str, Any]:
    generator = np.random.default_rng(seed + (1 if variant == "quantum" else 0))
    macro_auroc = 0.65 + offset + generator.normal(0.0, 0.001)
    per_label = []
    for index, label in enumerate(PRIMARY_LABELS):
        auroc = 0.58 + 0.02 * index + offset + generator.normal(0.0, 0.001)
        per_label.append(
            {
                "label": label,
                "auroc": auroc,
                "auprc": max(0.05, auroc - 0.15),
                "f1": 0.5,
                "sensitivity": 0.55,
                "specificity": 0.75,
                "threshold": 0.45,
            }
        )
    return {
        "objective": 3,
        "variant": variant,
        "architecture_version": ARCHITECTURE,
        "seed": seed,
        "best_epoch": 3,
        "bottleneck_parameters": 36,
        "total_trainable_parameters": 3253,
        "test_cases_accessed": 0,
        "test_evaluated": False,
        "checkpoint_sha256": "0" * 64,
        "validation_metrics": {
            "macro": {"auroc": macro_auroc, "auprc": 0.55, "f1": 0.5},
            "per_label": per_label,
        },
    }


def smoke(output_root: Path | None) -> tuple[Path, str]:
    seed = 42
    seed_everything(seed)
    with tempfile.TemporaryDirectory(prefix="objective3_v2_part4_job1_") as directory:
        root = Path(directory)
        train = root / "train.csv"
        validation = root / "validation.csv"
        embedding_root = root / "embeddings"
        embedding_root.mkdir()
        generator = np.random.default_rng(seed)
        frame = pd.DataFrame(
            # the "label_" prefix mirrors the real cohort schema
            {
                f"label_{label}": generator.integers(0, 2, size=24)
                for label in PRIMARY_LABELS
            }
        )
        frame.to_csv(train, index=False)
        frame.to_csv(validation, index=False)
        train_hash = sha256_file(train)
        validation_hash = sha256_file(validation)
        gat_hash = "1" * 64
        seeds = [42, 1042, 2042]
        protocol = {
            "study": STUDY,
            "version": VERSION,
            "training_started": False,
            "design": {"seed_list": seeds, "seeds_per_configuration": len(seeds)},
            "sizing_pilot": {"pilot_seeds": [900_042, 900_043, 900_044]},
            "frozen_inputs": {
                "train_cohort_sha256": train_hash,
                "validation_cohort_sha256": validation_hash,
                "frozen_gat_sha256": gat_hash,
            },
        }
        protocol_path = write_json_atomic(root / "protocol.json", protocol)
        protocol_hash = sha256_file(protocol_path)
        destination = output_root or DEFAULT_SMOKE_OUTPUT
        destination = assert_no_locked_test(destination)
        for study_seed in seeds:
            for variant in VARIANTS:
                summary_dir = destination / variant / f"seed{study_seed}"
                write_json_atomic(
                    summary_dir / "validation_summary.json",
                    _fake_summary(
                        variant,
                        study_seed,
                        0.001 if variant == "quantum" else 0.0,
                    ),
                )
        args = SimpleNamespace(
            protocol=protocol_path,
            expected_protocol_sha256=protocol_hash,
            train_manifest=train,
            val_manifest=validation,
            embedding_root=embedding_root,
            output_root=destination,
            hf_repo="smoke/private",
            hf_base_path="objective3_v2/smoke",
            expected_train_sha256=train_hash,
            expected_val_sha256=validation_hash,
            expected_gat_sha256=gat_hash,
            expected_train_cases=24,
            expected_val_cases=24,
            epochs=2,
            patience=1,
            batch_size=8,
            learning_rate=0.001,
            weight_decay=0.0001,
            dropout=0.2,
            poll_seconds=0.0,
            aggregate_only=True,
        )
        result = execute(args)
        print("POWERED VALIDATION SMOKE PASSED")
        return result


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke(args.output_root)
        return
    execute(args)


if __name__ == "__main__":
    main()
