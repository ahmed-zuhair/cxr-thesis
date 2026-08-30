#!/usr/bin/env python3
"""Run locked Objective 6 v1.1 training with epoch-level private recovery."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from train_objective6_with_private_recovery import (
    restore,
    sha256,
    snapshot,
    stable_recovery,
    upload,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "279e4fe83da6d82afcbcce595b5596980ca970fae958a16264d3b3e5172eb1a1"
LOCK_SHA256 = "b840440da16023c0169eb3f32c0f4ce7a20ecfa34f8f6b6bfa8ef20511aa53e6"
V1_CHECKPOINT_SHA256 = "18aa4293195b77aaf04df1ba310431df83f75e51ee6aa5837ff43a48a8ec10d3"
VARIANT = "clinical_guided_multimodal_v1_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--enhancement-protocol", type=Path, required=True)
    parser.add_argument("--enhancement-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--maximum-length", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clinical-loss-weight", type=float, default=0.35)
    parser.add_argument("--repetition-loss-weight", type=float, default=0.02)
    parser.add_argument("--train-cases", type=int, default=0)
    parser.add_argument("--validation-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.variant = VARIANT
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    validate_manifest(
        args.train_manifest, "train",
        "278addf3c0a216bb206b4e4b79364f26bacbee977f3209e9275e2abbd8fda7d7",
        29283,
    )
    validate_manifest(
        args.val_manifest, "val",
        "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6",
        6280,
    )
    protected = {
        args.v1_checkpoint: V1_CHECKPOINT_SHA256,
        args.enhancement_protocol: PROTOCOL_SHA256,
        args.enhancement_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 v1.1 input changed: {path}")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    if not bool(api.model_info(args.hf_repo, token=token).private):
        raise RuntimeError("Objective 6 recovery repository must remain private")
    remote = set(api.list_repo_files(args.hf_repo, repo_type="model", token=token))
    restored = restore(api, hf_hub_download, args, remote)
    final_summary = args.output_dir / "validation_summary.json"
    if final_summary.is_file():
        summary = json.loads(final_summary.read_text(encoding="utf-8"))
        if summary.get("variant") != VARIANT or summary.get("test_evaluated") is not False:
            raise RuntimeError("Recovered Objective 6 v1.1 result is incompatible")
        print(json.dumps({
            "event": "final_enhancement_training_restored",
            "variant": VARIANT, "training_repeated": False,
            "test_evaluated": False,
        }))
        print("OBJECTIVE 6 V1.1 PRIVATE TRAINING RESULT RESTORED SUCCESSFULLY")
        return
    local_recovery = stable_recovery(args.output_dir) if args.output_dir.exists() else None
    if args.output_dir.exists() and not restored and local_recovery is None:
        raise RuntimeError("Existing Objective 6 v1.1 output has no stable recovery")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_objective6_enhanced_report_generator.py"),
        "--train-manifest", str(args.train_manifest),
        "--val-manifest", str(args.val_manifest),
        "--v1-checkpoint", str(args.v1_checkpoint),
        "--enhancement-protocol", str(args.enhancement_protocol),
        "--enhancement-lock", str(args.enhancement_lock),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--accumulation-steps", str(args.accumulation_steps),
        "--workers", str(args.workers), "--image-size", str(args.image_size),
        "--maximum-length", str(args.maximum_length),
        "--learning-rate", str(args.learning_rate),
        "--encoder-learning-rate", str(args.encoder_learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--clinical-loss-weight", str(args.clinical_loss_weight),
        "--repetition-loss-weight", str(args.repetition_loss_weight),
        "--train-cases", str(args.train_cases),
        "--validation-cases", str(args.validation_cases),
        "--seed", str(args.seed),
    ]
    if args.no_amp:
        command.append("--no-amp")
    if restored or local_recovery is not None:
        command.append("--resume")
    process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy())
    uploaded_epoch = local_recovery[1] if local_recovery else 0
    try:
        while process.poll() is None:
            stable = stable_recovery(args.output_dir) if args.output_dir.exists() else None
            if stable is not None and stable[1] > uploaded_epoch:
                with tempfile.TemporaryDirectory(
                    prefix="objective6_v1_1_recovery_"
                ) as directory:
                    paths, epoch = snapshot(args.output_dir, Path(directory))
                    if paths and epoch > uploaded_epoch:
                        upload(
                            api, CommitOperationAdd, args, paths,
                            f"recovery: Objective 6 v1.1 completed epoch {epoch}",
                        )
                        uploaded_epoch = epoch
                        print(json.dumps({
                            "private_recovery_uploaded_epoch": epoch,
                            "variant": VARIANT, "test_evaluated": False,
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
        raise RuntimeError("Final Objective 6 v1.1 artifacts are incomplete")
    if stable_recovery(args.output_dir) is None:
        raise RuntimeError("Final Objective 6 v1.1 recovery checkpoint is invalid")
    upload(
        api, CommitOperationAdd, args, final_paths,
        "recovery: finalize Objective 6 v1.1 validation candidate",
    )
    summary = json.loads(final_summary.read_text(encoding="utf-8"))
    print(json.dumps({
        "event": "private_enhancement_training_finalized",
        "variant": VARIANT, "best_epoch": summary["best_epoch"],
        "validation_total_joint_loss": summary["validation_total_joint_loss"],
        "validation_auxiliary_macro_f1": summary[
            "validation_auxiliary_macro_f1"
        ],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "test_evaluated": False, "private_recovery_verified": True,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 6 V1.1 TRAINING WITH PRIVATE RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()
