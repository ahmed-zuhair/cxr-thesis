#!/usr/bin/env python3
"""Train one Objective 6 variant with epoch-level private HF recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("image_only", "multimodal"), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--maximum-length", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(path: Path, split: str, expected_hash: str, cases: int) -> None:
    if not path.is_file() or sha256(path) != expected_hash:
        raise RuntimeError(f"Protected {split} report manifest hash mismatch")
    frame = pd.read_csv(path, usecols=["patient_id", "split"], low_memory=False)
    if len(frame) != cases or set(frame["split"].astype(str)) != {split}:
        raise RuntimeError(f"Protected {split} report manifest structure changed")


def stable_recovery(output: Path) -> tuple[str, int] | None:
    checkpoint = output / "last.pt"
    checksum = output / "last.pt.sha256"
    if not checkpoint.is_file() or not checksum.is_file():
        return None
    recorded = checksum.read_text(encoding="utf-8").split()[0]
    if sha256(checkpoint) != recorded:
        return None
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("test_evaluated") is not False:
        raise RuntimeError("Objective 6 recovery is not test-blind")
    return recorded, int(state["epoch_completed"])


def upload(api, operation, args: argparse.Namespace, paths: list[Path], message: str) -> None:
    prefix = args.hf_path.strip("/")
    api.create_commit(
        repo_id=args.hf_repo,
        repo_type="model",
        token=os.environ["HF_TOKEN"],
        operations=[
            operation(
                path_in_repo=f"{prefix}/{path.name}", path_or_fileobj=str(path)
            )
            for path in paths
        ],
        commit_message=message,
    )


def restore(api, download, args: argparse.Namespace, remote: set[str]) -> bool:
    prefix = args.hf_path.strip("/")
    names = (
        "last.pt", "last.pt.sha256", "best.pt", "best.pt.sha256",
        "history_progress.csv", "history.csv", "vocabulary.json",
        "vocabulary.json.sha256", "validation_summary.json",
    )
    available = [name for name in names if f"{prefix}/{name}" in remote]
    if not available:
        return False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in available:
        downloaded = Path(download(
            repo_id=args.hf_repo,
            filename=f"{prefix}/{name}",
            repo_type="model",
            token=os.environ["HF_TOKEN"],
            force_download=True,
        ))
        shutil.copy2(downloaded, args.output_dir / name)
    if (args.output_dir / "last.pt").is_file() and stable_recovery(args.output_dir) is None:
        raise RuntimeError("Downloaded Objective 6 recovery failed verification")
    return True


def snapshot(output: Path, target: Path) -> tuple[list[Path], int]:
    stable = stable_recovery(output)
    if stable is None:
        return [], 0
    names = (
        "last.pt", "last.pt.sha256", "best.pt", "best.pt.sha256",
        "history_progress.csv", "vocabulary.json", "vocabulary.json.sha256",
    )
    target.mkdir(parents=True)
    paths: list[Path] = []
    for name in names:
        source = output / name
        if source.is_file():
            destination = target / name
            shutil.copy2(source, destination)
            paths.append(destination)
    if stable_recovery(target) is None:
        return [], 0
    return paths, stable[1]


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    validate_manifest(args.train_manifest, "train", args.expected_train_sha256, 29283)
    validate_manifest(args.val_manifest, "val", args.expected_val_sha256, 6280)
    if sha256(args.source_checkpoint) != args.expected_source_sha256:
        raise RuntimeError("Objective 5 PadChest checkpoint hash mismatch")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    if not bool(api.model_info(args.hf_repo, token=token).private):
        raise RuntimeError("Objective 6 recovery repository must remain private")
    remote = set(api.list_repo_files(args.hf_repo, repo_type="model", token=token))
    restored = restore(api, hf_hub_download, args, remote)
    final_summary = args.output_dir / "validation_summary.json"
    if final_summary.is_file():
        summary = json.loads(final_summary.read_text(encoding="utf-8"))
        if summary.get("variant") != args.variant or summary.get("test_evaluated") is not False:
            raise RuntimeError("Recovered Objective 6 final result is incompatible")
        print(json.dumps({
            "event": "final_training_restored", "variant": args.variant,
            "training_repeated": False, "test_evaluated": False,
        }))
        print("OBJECTIVE 6 PRIVATE TRAINING RESULT RESTORED SUCCESSFULLY")
        return
    local_recovery = stable_recovery(args.output_dir) if args.output_dir.exists() else None
    if args.output_dir.exists() and not restored and local_recovery is None:
        raise RuntimeError("Existing Objective 6 output has no stable recovery")

    command = [
        sys.executable, str(ROOT / "scripts" / "train_objective6_report_generator.py"),
        "--variant", args.variant,
        "--train-manifest", str(args.train_manifest),
        "--val-manifest", str(args.val_manifest),
        "--source-checkpoint", str(args.source_checkpoint),
        "--expected-source-sha256", args.expected_source_sha256,
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--accumulation-steps", str(args.accumulation_steps),
        "--workers", str(args.workers),
        "--image-size", str(args.image_size),
        "--maximum-length", str(args.maximum_length),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--seed", str(args.seed),
    ]
    if args.no_amp:
        command.append("--no-amp")
    if restored or local_recovery is not None:
        command.append("--resume")

    process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy())
    uploaded_epoch = local_recovery[1] if restored and local_recovery else 0
    try:
        while process.poll() is None:
            stable = stable_recovery(args.output_dir) if args.output_dir.exists() else None
            if stable is not None and stable[1] > uploaded_epoch:
                with tempfile.TemporaryDirectory(prefix="objective6_recovery_") as directory:
                    paths, epoch = snapshot(args.output_dir, Path(directory))
                    if paths and epoch > uploaded_epoch:
                        upload(
                            api, CommitOperationAdd, args, paths,
                            f"recovery: Objective 6 {args.variant} completed epoch {epoch}",
                        )
                        uploaded_epoch = epoch
                        print(json.dumps({
                            "private_recovery_uploaded_epoch": epoch,
                            "variant": args.variant, "test_evaluated": False,
                        }), flush=True)
            time.sleep(args.poll_seconds)
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    final_names = (
        "last.pt", "last.pt.sha256", "best.pt", "best.pt.sha256",
        "history_progress.csv", "history.csv", "vocabulary.json",
        "vocabulary.json.sha256", "validation_summary.json",
    )
    final_paths = [args.output_dir / name for name in final_names]
    if not all(path.is_file() for path in final_paths):
        raise RuntimeError("Final Objective 6 training artifacts are incomplete")
    if stable_recovery(args.output_dir) is None:
        raise RuntimeError("Final Objective 6 recovery checkpoint is invalid")
    upload(
        api, CommitOperationAdd, args, final_paths,
        f"recovery: finalize Objective 6 {args.variant} validation candidate",
    )
    summary = json.loads(final_summary.read_text(encoding="utf-8"))
    print(json.dumps({
        "event": "private_training_finalized",
        "variant": args.variant,
        "best_epoch": summary["best_epoch"],
        "validation_loss": summary["validation_loss"],
        "validation_perplexity": summary["validation_perplexity"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "test_evaluated": False,
        "private_recovery_verified": True,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 6 TEST-BLIND TRAINING WITH PRIVATE RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()
