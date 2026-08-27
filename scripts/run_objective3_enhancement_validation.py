#!/usr/bin/env python3
"""Run and aggregate the six locked Objective 3 v1.1 validation experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("classical_matched", "quantum")
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--protocol-amendment", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-base-path", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-gat-sha256", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(path: Path, expected_hash: str) -> dict[str, object]:
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RuntimeError("Objective 3 v1.1 protocol amendment SHA-256 does not match")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "version": protocol.get("amendment_version") == "1.1",
        "architecture": protocol.get("frozen_enhancement", {}).get("architecture")
        == "v1_1_reupload_gated",
        "parameter_match": protocol.get("frozen_enhancement", {}).get(
            "parameter_matched"
        )
        is True,
        "test_cohort": protocol.get("new_test_cohort_selected") is False,
        "test_manifest": protocol.get("test_manifest_opened") is False,
        "test_labels": protocol.get("test_labels_accessed") is False,
        "test_evaluation": protocol.get("test_evaluated") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Objective 3 v1.1 protocol checks failed: {failed}")
    return protocol


def run_one(args: argparse.Namespace, variant: str, seed: int) -> dict[str, object]:
    output = args.output_root / variant / f"seed{seed}"
    remote = (
        f"{args.hf_base_path.strip('/')}/{variant}/seed{seed}/validation_v1.0.0"
    )
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "train_objective3_with_private_recovery.py"),
        "--variant",
        variant,
        "--architecture",
        "v1_1_reupload_gated",
        "--train-manifest",
        str(args.train_manifest),
        "--val-manifest",
        str(args.val_manifest),
        "--embedding-root",
        str(args.embedding_root),
        "--output-dir",
        str(output),
        "--hf-repo",
        args.hf_repo,
        "--hf-path",
        remote,
        "--expected-train-sha256",
        args.expected_train_sha256,
        "--expected-val-sha256",
        args.expected_val_sha256,
        "--expected-gat-sha256",
        args.expected_gat_sha256,
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
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    print("\n--- STARTING", variant, "SEED", seed, "---", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    summary_path = output / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "architecture": summary.get("architecture_version")
        == "v1_1_reupload_gated",
        "variant": summary.get("variant") == variant,
        "seed": summary.get("seed") == seed,
        "research": summary.get("research_result") is True,
        "test_cases": summary.get("test_cases_accessed") == 0,
        "test": summary.get("test_evaluated") is False,
        "bottleneck_parameters": summary.get("bottleneck_parameters") == 36,
        "total_parameters": summary.get("total_trainable_parameters") == 3253,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Completed run failed checks: {variant}/{seed}: {failed}")
    return {
        "variant": variant,
        "seed": seed,
        "best_epoch": summary["best_epoch"],
        "validation_macro_auroc": summary["validation_metrics"]["macro"]["auroc"],
        "validation_macro_auprc": summary["validation_metrics"]["macro"]["auprc"],
        "validation_macro_f1": summary["validation_metrics"]["macro"]["f1"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "private_recovery_path": remote,
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    validate_protocol(args.protocol_amendment, args.expected_protocol_sha256)
    args.output_root.mkdir(parents=True, exist_ok=True)
    runs = [
        run_one(args, variant, seed)
        for seed in SEEDS
        for variant in VARIANTS
    ]
    by_key = {(run["variant"], run["seed"]): run for run in runs}
    classical = np.asarray(
        [by_key[("classical_matched", seed)]["validation_macro_auroc"] for seed in SEEDS],
        dtype=np.float64,
    )
    quantum = np.asarray(
        [by_key[("quantum", seed)]["validation_macro_auroc"] for seed in SEEDS],
        dtype=np.float64,
    )
    differences = quantum - classical
    seed_wins = int(np.sum(differences > 0.0))
    mean_difference = float(differences.mean())
    advance = mean_difference > 0.0 and seed_wins >= 2
    aggregate = {
        "artifact": "Objective 3 v1.1 enhanced paired validation result",
        "objective": 3,
        "architecture_version": "v1_1_reupload_gated",
        "protocol_amendment_sha256": args.expected_protocol_sha256,
        "seeds": list(SEEDS),
        "runs": runs,
        "classical_mean_validation_macro_auroc": float(classical.mean()),
        "quantum_mean_validation_macro_auroc": float(quantum.mean()),
        "quantum_minus_classical_by_seed": differences.tolist(),
        "mean_quantum_minus_classical_validation_macro_auroc": mean_difference,
        "quantum_seed_wins": seed_wins,
        "required_seed_wins": 2,
        "advance_to_single_final_evaluation": advance,
        "additional_architecture_tuning_allowed": False,
        "new_test_cohort_selected": False,
        "test_manifest_opened": False,
        "test_labels_accessed": False,
        "test_evaluated": False,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "private_checkpoints_included": False,
        "allowed_for_publication": True,
    }
    summary_path = args.output_root / "objective3_enhancement_validation_summary_public.json"
    summary_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_hash = sha256_file(summary_path)
    summary_path.with_suffix(".json.sha256").write_text(
        f"{summary_hash}  {summary_path.name}\n", encoding="utf-8"
    )
    print("\n--- OBJECTIVE 3 V1.1 PAIRED VALIDATION RESULT ---")
    print("Classical mean validation AUROC:", float(classical.mean()))
    print("Quantum mean validation AUROC:", float(quantum.mean()))
    print("Quantum minus classical by seed:", differences.tolist())
    print("Mean quantum minus classical AUROC:", mean_difference)
    print("Quantum seed wins:", seed_wins, "of 3")
    print("Advance to single final evaluation:", advance)
    print("Summary SHA-256:", summary_hash)
    print("Test manifest opened: False")
    print("Test labels accessed: False")
    print("Test evaluated: False")
    print("OBJECTIVE 3 V1.1 SIX PAIRED VALIDATION RUNS SUCCESSFUL")


if __name__ == "__main__":
    main()
