#!/usr/bin/env python3
"""Run one paired Objective 3 head with epoch-level private HF recovery."""

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

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a test-blind Objective 3 head with private recovery"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=("quantum", "classical_matched"),
    )
    parser.add_argument(
        "--architecture",
        choices=("v1_concat", "v1_1_reupload_gated"),
        default="v1_concat",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-gat-sha256", required=True)
    parser.add_argument("--expected-train-cases", type=int, default=30_000)
    parser.add_argument("--expected-val-cases", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_is_test_blind(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("test_evaluated") is not False:
        raise RuntimeError(f"Checkpoint is not test-blind: {path}")
    return int(state.get("epoch_completed", state.get("epoch", 0)))


def stable_recovery(output_dir: Path) -> tuple[str, int] | None:
    checkpoint = output_dir / "last.pt"
    checksum = output_dir / "last.sha256"
    if not checkpoint.is_file() or not checksum.is_file():
        return None
    recorded = checksum.read_text(encoding="utf-8").split()[0]
    if sha256_file(checkpoint) != recorded:
        return None
    return recorded, checkpoint_is_test_blind(checkpoint)


def restore_files(
    hf_hub_download,
    args: argparse.Namespace,
    remote_files: set[str],
) -> bool:
    prefix = args.hf_path.strip("/")
    final_remote = f"{prefix}/validation_summary.json"
    recovery_remote = f"{prefix}/last.pt"
    if final_remote not in remote_files and recovery_remote not in remote_files:
        return False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "last.pt",
        "last.sha256",
        "history_progress.csv",
        "best.pt",
        "best.sha256",
        "history.csv",
        "validation_summary.json",
    )
    for name in names:
        remote = f"{prefix}/{name}"
        if remote not in remote_files:
            continue
        downloaded = Path(
            hf_hub_download(
                args.hf_repo,
                filename=remote,
                repo_type="model",
                token=os.environ["HF_TOKEN"],
                local_dir=args.output_dir / ".downloads",
                force_download=True,
            )
        )
        shutil.copy2(downloaded, args.output_dir / name)
    if (args.output_dir / "last.pt").is_file() and stable_recovery(
        args.output_dir
    ) is None:
        raise RuntimeError("Downloaded last.pt failed SHA-256 verification")
    if (args.output_dir / "best.sha256").is_file():
        recorded = (
            (args.output_dir / "best.sha256").read_text(encoding="utf-8").split()[0]
        )
        if sha256_file(args.output_dir / "best.pt") != recorded:
            raise RuntimeError("Downloaded best.pt failed SHA-256 verification")
    return True


def snapshot_recovery(
    output_dir: Path, destination: Path
) -> tuple[list[Path], int]:
    stable = stable_recovery(output_dir)
    if stable is None:
        return [], 0
    expected_hash, epoch = stable
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    for name in ("last.pt", "last.sha256", "history_progress.csv", "best.pt"):
        source = output_dir / name
        if source.is_file():
            target = destination / name
            shutil.copy2(source, target)
            selected.append(target)
    if sha256_file(destination / "last.pt") != expected_hash:
        return [], 0
    checkpoint_is_test_blind(destination / "last.pt")
    return selected, epoch


def upload_paths(
    api,
    CommitOperationAdd,
    args: argparse.Namespace,
    paths: list[Path],
    message: str,
) -> None:
    prefix = args.hf_path.strip("/")
    api.create_commit(
        repo_id=args.hf_repo,
        repo_type="model",
        token=os.environ["HF_TOKEN"],
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{prefix}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in paths
        ],
        commit_message=message,
    )


def validate_final_summary(path: Path, args: argparse.Namespace) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    configuration = summary.get("training_configuration", {})
    checks = {
        "objective": summary.get("objective") == 3,
        "variant": summary.get("variant") == args.variant,
        "architecture": summary.get("architecture_version") == args.architecture,
        "seed": summary.get("seed") == args.seed,
        "train_hash": configuration.get("train_manifest_sha256")
        == args.expected_train_sha256,
        "validation_hash": configuration.get("validation_manifest_sha256")
        == args.expected_val_sha256,
        "gat_hash": configuration.get("gat_checkpoint_sha256")
        == args.expected_gat_sha256,
        "limit_train": configuration.get("limit_train") == args.limit_train,
        "limit_val": configuration.get("limit_val") == args.limit_val,
        "test_cases": summary.get("test_cases_accessed") == 0,
        "test_evaluated": summary.get("test_evaluated") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Recovered final summary failed checks: {failed}")
    return summary


def main() -> None:
    args = parse_args()
    if not os.environ.get("HF_TOKEN", "").strip():
        raise RuntimeError("HF_TOKEN is not loaded")
    if sha256_file(args.train_manifest) != args.expected_train_sha256:
        raise RuntimeError("Training manifest SHA-256 does not match")
    if sha256_file(args.val_manifest) != args.expected_val_sha256:
        raise RuntimeError("Validation manifest SHA-256 does not match")
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub before private recovery") from error
    api = HfApi(token=os.environ["HF_TOKEN"])
    info = api.model_info(args.hf_repo, token=os.environ["HF_TOKEN"])
    if not bool(info.private):
        raise RuntimeError("Objective 3 training recovery repository must be private")
    remote_files = set(
        api.list_repo_files(
            args.hf_repo,
            repo_type="model",
            token=os.environ["HF_TOKEN"],
        )
    )
    restored = restore_files(hf_hub_download, args, remote_files)
    final_summary = args.output_dir / "validation_summary.json"
    if final_summary.is_file():
        summary = validate_final_summary(final_summary, args)
        final_names = (
            "best.pt",
            "best.sha256",
            "history.csv",
            "validation_summary.json",
            "last.pt",
            "last.sha256",
            "history_progress.csv",
        )
        final_paths = [args.output_dir / name for name in final_names]
        if not all(path.is_file() for path in final_paths):
            raise RuntimeError("Recovered final Objective 3 artifacts are incomplete")
        expected_best = (
            (args.output_dir / "best.sha256")
            .read_text(encoding="utf-8")
            .split()[0]
        )
        if sha256_file(args.output_dir / "best.pt") != expected_best:
            raise RuntimeError("Recovered final best.pt checksum does not match")
        remote_final = f"{args.hf_path.strip('/')}/validation_summary.json"
        if remote_final not in remote_files:
            upload_paths(
                api,
                CommitOperationAdd,
                args,
                final_paths,
                f"recovery: finalize Objective 3 {args.variant} seed {args.seed}",
            )
        print(
            json.dumps(
                {
                    "event": "objective3_final_training_restored",
                    "variant": args.variant,
                    "seed": args.seed,
                    "best_epoch": summary["best_epoch"],
                    "validation_macro_auroc": summary["validation_metrics"]["macro"][
                        "auroc"
                    ],
                    "training_repeated": False,
                    "final_private_recovery_verified": True,
                    "test_evaluated": False,
                },
                indent=2,
            )
        )
        print("OBJECTIVE 3 PRIVATE TRAINING RESULT RESTORED SUCCESSFULLY")
        return

    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "train_objective3_head.py"),
        "--variant",
        args.variant,
        "--architecture",
        args.architecture,
        "--train-manifest",
        str(args.train_manifest),
        "--val-manifest",
        str(args.val_manifest),
        "--embedding-root",
        str(args.embedding_root),
        "--output-dir",
        str(args.output_dir),
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
        str(args.seed),
    ]
    if args.limit_train is not None:
        command.extend(("--limit-train", str(args.limit_train)))
    if args.limit_val is not None:
        command.extend(("--limit-val", str(args.limit_val)))
    if restored or stable_recovery(args.output_dir) is not None:
        command.append("--resume")

    process = subprocess.Popen(command)
    uploaded_epoch = 0
    try:
        while process.poll() is None:
            stable = stable_recovery(args.output_dir)
            if stable is not None and stable[1] > uploaded_epoch:
                with tempfile.TemporaryDirectory(
                    prefix="objective3_recovery_"
                ) as directory:
                    paths, epoch = snapshot_recovery(
                        args.output_dir, Path(directory)
                    )
                    if paths and epoch > uploaded_epoch:
                        upload_paths(
                            api,
                            CommitOperationAdd,
                            args,
                            paths,
                            (
                                "recovery: Objective 3 "
                                f"{args.variant} seed {args.seed} epoch {epoch}"
                            ),
                        )
                        uploaded_epoch = epoch
                        print(
                            json.dumps(
                                {
                                    "private_recovery_uploaded_epoch": epoch,
                                    "variant": args.variant,
                                    "architecture_version": args.architecture,
                                    "seed": args.seed,
                                    "test_evaluated": False,
                                }
                            ),
                            flush=True,
                        )
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
        "best.pt",
        "best.sha256",
        "history.csv",
        "validation_summary.json",
        "last.pt",
        "last.sha256",
        "history_progress.csv",
    )
    final_paths = [args.output_dir / name for name in final_names]
    if not all(path.is_file() for path in final_paths):
        raise RuntimeError("Final Objective 3 training artifacts are incomplete")
    expected_best = (
        (args.output_dir / "best.sha256").read_text(encoding="utf-8").split()[0]
    )
    if sha256_file(args.output_dir / "best.pt") != expected_best:
        raise RuntimeError("Final best.pt checksum does not match")
    checkpoint_is_test_blind(args.output_dir / "best.pt")
    summary = validate_final_summary(final_summary, args)
    upload_paths(
        api,
        CommitOperationAdd,
        args,
        final_paths,
        f"recovery: finalize Objective 3 {args.variant} seed {args.seed}",
    )
    print(
        json.dumps(
            {
                "event": "objective3_private_training_finalized",
                "variant": args.variant,
                "architecture_version": args.architecture,
                "seed": args.seed,
                "best_epoch": summary["best_epoch"],
                "validation_macro_auroc": summary["validation_metrics"]["macro"][
                    "auroc"
                ],
                "validation_macro_auprc": summary["validation_metrics"]["macro"][
                    "auprc"
                ],
                "checkpoint_sha256": expected_best,
                "test_evaluated": False,
                "private_recovery_verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 3 TEST-BLIND PAIRED TRAINING WITH PRIVATE RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()
