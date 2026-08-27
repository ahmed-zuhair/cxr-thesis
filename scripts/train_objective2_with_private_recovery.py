#!/usr/bin/env python3
"""Run one Objective 2 candidate with epoch-level private HF recovery."""

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective2.graph_generation import safe_graph_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a test-blind Objective 2 candidate with private recovery"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("cnn", "attention_cnn", "vit", "gcn", "gat", "densenet121"),
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-train-cases", type=int, default=30_000)
    parser.add_argument("--expected-val-cases", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--augmentation-profile", choices=("baseline", "cxr_mild"), default="baseline"
    )
    parser.add_argument("--epoch-varying-augmentation", action="store_true")
    parser.add_argument("--loss", choices=("bce", "asymmetric"), default="bce")
    parser.add_argument(
        "--positive-weight-transform",
        choices=("raw", "sqrt", "log1p", "none"),
        default="raw",
    )
    parser.add_argument("--max-positive-weight", type=float)
    parser.add_argument("--scheduler", choices=("plateau", "cosine"), default="plateau")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--backbone-learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(
    path: Path, split: str, expected_hash: str, cases: int
) -> pd.DataFrame:
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"{split} manifest SHA-256 does not match")
    frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str})
    if len(frame) != cases:
        raise RuntimeError(f"{split} manifest has {len(frame)} cases, expected {cases}")
    if set(frame["split"].astype(str).str.lower()) != {split}:
        raise RuntimeError(f"{split} manifest contains another split")
    return frame


def checkpoint_is_test_blind(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("test_evaluated") is not False:
        raise RuntimeError(f"Recovery checkpoint is not test-blind: {path}")
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
    api, hf_hub_download, args: argparse.Namespace, remote_files: set[str]
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
    if (args.output_dir / "last.pt").is_file():
        recovery = stable_recovery(args.output_dir)
        if recovery is None:
            raise RuntimeError(
                "Downloaded recovery checkpoint failed SHA-256 verification"
            )
    if (args.output_dir / "best.sha256").is_file():
        expected = (
            (args.output_dir / "best.sha256").read_text(encoding="utf-8").split()[0]
        )
        if sha256_file(args.output_dir / "best.pt") != expected:
            raise RuntimeError("Downloaded final best.pt failed SHA-256 verification")
    return True


def snapshot_recovery(output_dir: Path, destination: Path) -> tuple[list[Path], int]:
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
        shutil.rmtree(destination)
        return [], 0
    checkpoint_is_test_blind(destination / "last.pt")
    return selected, epoch


def upload_paths(
    api, CommitOperationAdd, args: argparse.Namespace, paths: list[Path], message: str
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


def main() -> None:
    args = parse_args()
    if not os.environ.get("HF_TOKEN", "").strip():
        raise RuntimeError("HF_TOKEN is not loaded")
    train = validate_manifest(
        args.train_manifest,
        "train",
        args.expected_train_sha256,
        args.expected_train_cases,
    )
    validation = validate_manifest(
        args.val_manifest,
        "val",
        args.expected_val_sha256,
        args.expected_val_cases,
    )
    if set(train["patient_id"]) & set(validation["patient_id"]):
        raise RuntimeError("Patient leakage exists between training and validation")
    if args.model in {"gcn", "gat"}:
        if args.graph_root is None:
            raise ValueError("Graph root is required for a graph model")
        expected_graphs = {
            f"{safe_graph_name(value)}.npz"
            for value in pd.concat([train, validation])["image_id"].astype(str)
        }
        actual_graphs = {path.name for path in args.graph_root.glob("*.npz")}
        if actual_graphs != expected_graphs:
            raise RuntimeError(
                f"Graph root mismatch: expected {len(expected_graphs)}, found {len(actual_graphs)}"
            )

    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub before private recovery") from error
    api = HfApi(token=os.environ["HF_TOKEN"])
    info = api.model_info(args.hf_repo, token=os.environ["HF_TOKEN"])
    if not bool(info.private):
        raise RuntimeError("Training recovery repository must be private")
    remote_files = set(
        api.list_repo_files(
            args.hf_repo,
            repo_type="model",
            token=os.environ["HF_TOKEN"],
        )
    )
    restored = restore_files(api, hf_hub_download, args, remote_files)
    final_summary = args.output_dir / "validation_summary.json"
    if final_summary.is_file():
        summary = json.loads(final_summary.read_text(encoding="utf-8"))
        if (
            summary.get("model") != args.model
            or summary.get("test_evaluated") is not False
        ):
            raise RuntimeError(
                "Recovered final summary does not match this test-blind run"
            )
        print(
            json.dumps(
                {
                    "event": "final_training_restored",
                    "model": args.model,
                    "training_repeated": False,
                    "test_evaluated": False,
                }
            )
        )
        print("OBJECTIVE 2 PRIVATE TRAINING RESULT RESTORED SUCCESSFULLY")
        return

    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "train_objective2_classifier.py"),
        "--model",
        args.model,
        "--train-manifest",
        str(args.train_manifest),
        "--val-manifest",
        str(args.val_manifest),
        "--output-dir",
        str(args.output_dir),
        "--data-root",
        str(args.data_root),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--image-size",
        str(args.image_size),
        "--seed",
        str(args.seed),
        "--dropout",
        str(args.dropout),
        "--augmentation-profile",
        args.augmentation_profile,
        "--loss",
        args.loss,
        "--positive-weight-transform",
        args.positive_weight_transform,
        "--scheduler",
        args.scheduler,
        "--accumulation-steps",
        str(args.accumulation_steps),
        "--backbone-learning-rate-multiplier",
        str(args.backbone_learning_rate_multiplier),
    ]
    if args.pretrained:
        command.append("--pretrained")
    if args.epoch_varying_augmentation:
        command.append("--epoch-varying-augmentation")
    if args.max_positive_weight is not None:
        command.extend(["--max-positive-weight", str(args.max_positive_weight)])
    if args.gradient_clip_norm is not None:
        command.extend(["--gradient-clip-norm", str(args.gradient_clip_norm)])
    if args.graph_root is not None:
        command.extend(["--graph-root", str(args.graph_root)])
    if args.no_amp:
        command.append("--no-amp")
    if args.limit_train is not None:
        command.extend(["--limit-train", str(args.limit_train)])
    if args.limit_val is not None:
        command.extend(["--limit-val", str(args.limit_val)])
    if restored or stable_recovery(args.output_dir) is not None:
        command.append("--resume")

    process = subprocess.Popen(command)
    uploaded_epoch = 0
    try:
        while process.poll() is None:
            stable = stable_recovery(args.output_dir)
            if stable is not None and stable[1] > uploaded_epoch:
                with tempfile.TemporaryDirectory(
                    prefix="objective2_recovery_"
                ) as directory:
                    paths, epoch = snapshot_recovery(
                        args.output_dir,
                        Path(directory),
                    )
                    if paths and epoch > uploaded_epoch:
                        upload_paths(
                            api,
                            CommitOperationAdd,
                            args,
                            paths,
                            f"recovery: {args.model} completed epoch {epoch}",
                        )
                        uploaded_epoch = epoch
                        print(
                            json.dumps(
                                {
                                    "private_recovery_uploaded_epoch": epoch,
                                    "model": args.model,
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
        raise RuntimeError("Final training artifacts are incomplete")
    expected_best = (
        (args.output_dir / "best.sha256").read_text(encoding="utf-8").split()[0]
    )
    if sha256_file(args.output_dir / "best.pt") != expected_best:
        raise RuntimeError("Final best.pt checksum does not match")
    checkpoint_is_test_blind(args.output_dir / "best.pt")
    upload_paths(
        api,
        CommitOperationAdd,
        args,
        final_paths,
        f"recovery: finalize Objective 2 {args.model} validation candidate",
    )
    summary = json.loads(final_summary.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "event": "private_training_finalized",
                "model": args.model,
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
    print("OBJECTIVE 2 TEST-BLIND TRAINING WITH PRIVATE RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()
