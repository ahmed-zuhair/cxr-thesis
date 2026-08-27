#!/usr/bin/env python3
"""Run all Objective 4 quantitative-XAI shards with private HF recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


COHORT_SHA256 = "daa7eeda7104f64dcd353f45604310748ca2ff84ea9ffa7cb4110e7c8daa0d2a"
CLASSIFIER_SHA256 = "2b7fa0d2f3dee3c59c538be15dd0435c71ad26b411fc1312bd7e5fe99fbac55f"
SEGMENTATION_SHA256 = "6ee1b4d351fdcfaaeec5e0487198128a5540d6dfe69a79a3158318aa22d9984c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--segmentation-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/"))
    parser.add_argument(
        "--public-repository", default="ahmed-zuhair/cxr-thesis-checkpoints"
    )
    parser.add_argument(
        "--private-repository", default="ahmed-zuhair/cxr-thesis-private-recovery"
    )
    parser.add_argument(
        "--private-cohort-path",
        default=(
            "objective4/xai/protocol/seed42/v1.0.0/private/"
            "xai_validation_cohort_private.csv"
        ),
    )
    parser.add_argument(
        "--public-classifier-path",
        default=(
            "objective2/classification/densenet121/seed42/"
            "validation_candidate_v1.0.0/best.pt"
        ),
    )
    parser.add_argument(
        "--public-segmentation-path",
        default=(
            "objective1/final/seed42/v1.0.0/checkpoint/"
            "adapted_roi_segmentation_best.pt"
        ),
    )
    parser.add_argument(
        "--private-recovery-path",
        default="objective4/xai/quantitative/seed42/v1.0.0",
    )
    parser.add_argument("--shard-size", type=int, default=12)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--ig-internal-batch-size", type=int, default=8)
    parser.add_argument("--faithfulness-steps", type=int, default=11)
    parser.add_argument("--stability-gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_hf_file(
    *,
    repository: str,
    remote_path: str,
    destination: Path,
    expected_hash: str,
    token: str,
) -> str:
    if destination.is_file() and sha256_file(destination) == expected_hash:
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=repository,
            filename=remote_path,
            repo_type="model",
            token=token,
            force_download=True,
        )
    )
    if sha256_file(downloaded) != expected_hash:
        raise RuntimeError(f"Downloaded artifact hash mismatch: {remote_path}")
    temporary = destination.with_name(f".{destination.name}.download")
    shutil.copy2(downloaded, temporary)
    os.replace(temporary, destination)
    return "restored"


def shard_paths(root: Path, index: int) -> dict[str, Path]:
    prefix = f"shard_{index:03d}"
    shard = root / "private" / "shards" / prefix
    return {
        "root": shard,
        "metrics": shard / f"{prefix}_metrics_private.csv",
        "saliency": shard / f"{prefix}_saliency_private.npz",
        "summary": shard / f"{prefix}_summary_private.json",
        "checksum": shard / f"{prefix}_summary_private.sha256",
    }


def validate_shard(root: Path, index: int, expected_cases: int) -> dict[str, object]:
    paths = shard_paths(root, index)
    missing = [name for name, path in paths.items() if name != "root" and not path.is_file()]
    if missing:
        raise RuntimeError(f"Shard {index:03d} files are missing: {missing}")
    recorded = paths["checksum"].read_text(encoding="utf-8").split()[0]
    actual_summary_hash = sha256_file(paths["summary"])
    if recorded != actual_summary_hash:
        raise RuntimeError(f"Shard {index:03d} summary checksum mismatch")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    checks = {
        "index": int(summary.get("shard_index", -1)) == index,
        "cases": int(summary.get("cases", -1)) == expected_cases,
        "rows": int(summary.get("metric_rows", -1)) == expected_cases * 2,
        "cohort": summary.get("cohort_sha256") == COHORT_SHA256,
        "classifier": summary.get("classifier_sha256") == CLASSIFIER_SHA256,
        "segmentation": summary.get("segmentation_sha256") == SEGMENTATION_SHA256,
        "metrics_hash": summary.get("metrics_sha256") == sha256_file(paths["metrics"]),
        "saliency_hash": summary.get("saliency_sha256") == sha256_file(paths["saliency"]),
        "test_blind": summary.get("test_evaluated") is False,
        "no_images": summary.get("medical_images_saved") is False,
        "no_masks": summary.get("predicted_masks_saved") is False,
        "private": summary.get("allowed_for_public_upload") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Shard {index:03d} integrity checks failed: {checks}")
    metrics = pd.read_csv(paths["metrics"])
    if len(metrics) != expected_cases * 2 or set(metrics["method"]) != {
        "grad_cam", "integrated_gradients"
    }:
        raise RuntimeError(f"Shard {index:03d} metric rows are invalid")
    with np.load(paths["saliency"], allow_pickle=False) as maps:
        if maps["grad_cam"].shape != (expected_cases, 320, 320):
            raise RuntimeError(f"Shard {index:03d} Grad-CAM shape is invalid")
        if maps["integrated_gradients"].shape != (expected_cases, 320, 320):
            raise RuntimeError(f"Shard {index:03d} IG shape is invalid")
        if len(maps["image_id"]) != expected_cases:
            raise RuntimeError(f"Shard {index:03d} private identity count is invalid")
    return summary


def restore_remote_shard(
    *,
    api: HfApi,
    repository: str,
    remote_root: str,
    output_root: Path,
    index: int,
    expected_cases: int,
    remote_files: set[str],
    token: str,
) -> bool:
    paths = shard_paths(output_root, index)
    prefix = f"shard_{index:03d}"
    remote_shard = f"{remote_root}/shards/{prefix}"
    names = {
        "metrics": f"{prefix}_metrics_private.csv",
        "saliency": f"{prefix}_saliency_private.npz",
        "summary": f"{prefix}_summary_private.json",
        "checksum": f"{prefix}_summary_private.sha256",
    }
    if not all(f"{remote_shard}/{name}" in remote_files for name in names.values()):
        return False
    if paths["root"].exists():
        shutil.rmtree(paths["root"])
    paths["root"].mkdir(parents=True)
    for key, name in names.items():
        downloaded = hf_hub_download(
            repo_id=repository,
            filename=f"{remote_shard}/{name}",
            repo_type="model",
            token=token,
        )
        shutil.copy2(downloaded, paths[key])
    validate_shard(output_root, index, expected_cases)
    return True


def upload_shard(
    *, repository: str, remote_root: str, shard_root: Path, index: int
) -> None:
    remote = f"{remote_root}/shards/shard_{index:03d}"
    command = [
        "hf", "upload", repository, str(shard_root), remote,
        "--repo-type", "model",
        "--commit-message", f"recovery: Objective 4 XAI shard {index:03d}",
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Private HF upload failed for shard {index:03d}")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    if args.shard_size <= 0 or 240 % args.shard_size != 0:
        raise ValueError("Shard size must divide the locked 240-case cohort")
    if not args.repository.is_dir():
        raise FileNotFoundError(args.repository)
    if not args.segmentation_config.is_file():
        raise FileNotFoundError(args.segmentation_config)

    api = HfApi(token=token)
    public_info = api.repo_info(args.public_repository, repo_type="model", token=token)
    private_info = api.repo_info(args.private_repository, repo_type="model", token=token)
    if bool(public_info.private):
        raise RuntimeError("The public checkpoint repository is unexpectedly private")
    if not bool(private_info.private):
        raise RuntimeError("The private recovery repository must remain private")

    print("--- RESTORING PROTECTED OBJECTIVE 4 INPUTS ---")
    actions = {
        "cohort": restore_hf_file(
            repository=args.private_repository,
            remote_path=args.private_cohort_path,
            destination=args.cohort,
            expected_hash=COHORT_SHA256,
            token=token,
        ),
        "classifier": restore_hf_file(
            repository=args.public_repository,
            remote_path=args.public_classifier_path,
            destination=args.classifier_checkpoint,
            expected_hash=CLASSIFIER_SHA256,
            token=token,
        ),
        "segmentation": restore_hf_file(
            repository=args.public_repository,
            remote_path=args.public_segmentation_path,
            destination=args.segmentation_checkpoint,
            expected_hash=SEGMENTATION_SHA256,
            token=token,
        ),
    }
    print("Input actions:", actions)
    cohort = pd.read_csv(args.cohort)
    if len(cohort) != 240:
        raise RuntimeError("Recovered Objective 4 cohort is not 240 cases")

    total_shards = 240 // args.shard_size
    remote_files = set(
        api.list_repo_files(args.private_repository, repo_type="model", token=token)
    )
    restored = 0
    generated = 0
    durations: list[float] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("--- OBJECTIVE 4 PRIVATE SHARD RECOVERY ---")
    print("Total cases:", 240)
    print("Total shards:", total_shards)
    print("Cases per shard:", args.shard_size)
    print("Private recovery path:", args.private_recovery_path)

    for index in range(total_shards):
        expected_cases = min(args.shard_size, 240 - index * args.shard_size)
        local = shard_paths(args.output_dir, index)
        try:
            validate_shard(args.output_dir, index, expected_cases)
            print(f"Shard {index:03d}: verified locally; skipped")
            restored += 1
            continue
        except (FileNotFoundError, RuntimeError, ValueError, KeyError):
            pass
        if restore_remote_shard(
            api=api,
            repository=args.private_repository,
            remote_root=args.private_recovery_path,
            output_root=args.output_dir,
            index=index,
            expected_cases=expected_cases,
            remote_files=remote_files,
            token=token,
        ):
            print(f"Shard {index:03d}: restored and verified from private HF")
            restored += 1
            continue
        if local["root"].exists():
            shutil.rmtree(local["root"])
        command = [
            sys.executable,
            str(args.repository / "scripts/run_objective4_xai_shard.py"),
            "--cohort", str(args.cohort),
            "--expected-cohort-sha256", COHORT_SHA256,
            "--classifier-checkpoint", str(args.classifier_checkpoint),
            "--expected-classifier-sha256", CLASSIFIER_SHA256,
            "--segmentation-checkpoint", str(args.segmentation_checkpoint),
            "--expected-segmentation-sha256", SEGMENTATION_SHA256,
            "--segmentation-config", str(args.segmentation_config),
            "--output-dir", str(args.output_dir),
            "--data-root", str(args.data_root),
            "--shard-index", str(index),
            "--shard-size", str(args.shard_size),
            "--ig-steps", str(args.ig_steps),
            "--ig-internal-batch-size", str(args.ig_internal_batch_size),
            "--faithfulness-steps", str(args.faithfulness_steps),
            "--stability-gamma", str(args.stability_gamma),
            "--seed", str(args.seed),
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=args.repository, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Objective 4 shard {index:03d} failed")
        summary = validate_shard(args.output_dir, index, expected_cases)
        upload_shard(
            repository=args.private_repository,
            remote_root=args.private_recovery_path,
            shard_root=local["root"],
            index=index,
        )
        refreshed = set(
            api.list_repo_files(args.private_repository, repo_type="model", token=token)
        )
        prefix = f"{args.private_recovery_path}/shards/shard_{index:03d}/"
        required = {
            prefix + f"shard_{index:03d}_metrics_private.csv",
            prefix + f"shard_{index:03d}_saliency_private.npz",
            prefix + f"shard_{index:03d}_summary_private.json",
            prefix + f"shard_{index:03d}_summary_private.sha256",
        }
        if not required.issubset(refreshed):
            raise RuntimeError(f"Private HF verification failed for shard {index:03d}")
        remote_files = refreshed
        duration = time.perf_counter() - started
        durations.append(duration)
        generated += 1
        remaining = total_shards - index - 1
        eta_minutes = (sum(durations) / len(durations)) * remaining / 60.0
        print(
            f"Shard {index:03d}: computed, verified, and privately backed up; "
            f"compute_seconds={summary['elapsed_seconds']:.1f}; "
            f"estimated_remaining_minutes={eta_minutes:.1f}"
        )

    for index in range(total_shards):
        validate_shard(args.output_dir, index, args.shard_size)
    print("--- FINAL OBJECTIVE 4 QUANTITATIVE-XAI SHARD STATUS ---")
    print("Verified cases:", 240)
    print("Verified shards:", total_shards)
    print("Restored/skipped shards:", restored)
    print("Newly generated shards:", generated)
    print("Grad-CAM maps:", 240)
    print("Integrated Gradients maps:", 240)
    print("Private recovery verified:", True)
    print("Medical images saved:", False)
    print("Predicted masks saved:", False)
    print("Test manifest opened:", False)
    print("Test labels accessed:", False)
    print("Test evaluated:", False)
    print("Allowed for public upload:", False)
    print("OBJECTIVE 4 FULL PRIVATE QUANTITATIVE-XAI SHARDS SUCCESSFUL")


if __name__ == "__main__":
    main()
