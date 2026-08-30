#!/usr/bin/env python3
"""Generate Objective 6 validation shards with private HF recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


VALIDATION_MANIFEST_SHA256 = (
    "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
)
VALIDATION_PROTOCOL_SHA256 = (
    "81424c30f1619707325f0a83ef9a6fba3a859743e3b4ee0c33ac68dba6161438"
)
VALIDATION_LOCK_SHA256 = (
    "e48b11cc0af8be0866b873ae91dd5f4c55738b39927d6dec52d2f29cf5f8275a"
)
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("image_only", "multimodal"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-protocol", type=Path, required=True)
    parser.add_argument("--validation-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--shard-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--maximum-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_shard(directory: Path, variant: str, index: int, count: int) -> bool:
    predictions = directory / "predictions_private.csv"
    checksum = predictions.with_suffix(".csv.sha256")
    summary_path = directory / "shard_summary_private.json"
    summary_checksum = summary_path.with_suffix(".json.sha256")
    if not all(path.is_file() for path in (
        predictions, checksum, summary_path, summary_checksum
    )):
        return False
    if sha256(predictions) != checksum.read_text(encoding="utf-8").split()[0]:
        return False
    if sha256(summary_path) != summary_checksum.read_text(encoding="utf-8").split()[0]:
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return all((
        summary.get("variant") == variant,
        summary.get("shard_index") == index,
        summary.get("shard_count") == count,
        summary.get("predictions_sha256") == sha256(predictions),
        summary.get("validation_manifest_sha256") == VALIDATION_MANIFEST_SHA256,
        summary.get("validation_protocol_sha256") == VALIDATION_PROTOCOL_SHA256,
        summary.get("validation_lock_sha256") == VALIDATION_LOCK_SHA256,
        summary.get("test_evaluated") is False,
        summary.get("public_upload_allowed") is False,
    ))


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    protected = {
        args.validation_manifest: VALIDATION_MANIFEST_SHA256,
        args.validation_protocol: VALIDATION_PROTOCOL_SHA256,
        args.validation_lock: VALIDATION_LOCK_SHA256,
        args.checkpoint: args.expected_checkpoint_sha256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 input changed: {path}")
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    if not bool(api.model_info(args.hf_repo, token=token).private):
        raise RuntimeError("Objective 6 validation recovery repository must be private")
    remote = set(api.list_repo_files(args.hf_repo, repo_type="model", token=token))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.hf_path.strip("/")
    required_names = (
        "predictions_private.csv", "predictions_private.csv.sha256",
        "shard_summary_private.json", "shard_summary_private.json.sha256",
    )
    actions: dict[str, int] = {"restored": 0, "generated": 0, "reused": 0}
    for index in range(args.shard_count):
        name = f"shard_{index:03d}"
        local = args.output_dir / "shards" / name
        if valid_shard(local, args.variant, index, args.shard_count):
            actions["reused"] += 1
            continue
        remote_root = f"{prefix}/shards/{name}"
        remote_names = {f"{remote_root}/{item}" for item in required_names}
        available = remote_names & remote
        if available and available != remote_names:
            raise RuntimeError(f"Partial remote Objective 6 shard exists: {name}")
        if available == remote_names:
            local.mkdir(parents=True, exist_ok=True)
            for filename in required_names:
                downloaded = Path(hf_hub_download(
                    repo_id=args.hf_repo, filename=f"{remote_root}/{filename}",
                    repo_type="model", token=token, force_download=True,
                ))
                shutil.copy2(downloaded, local / filename)
            if not valid_shard(local, args.variant, index, args.shard_count):
                raise RuntimeError(f"Remote Objective 6 shard failed verification: {name}")
            actions["restored"] += 1
            print(json.dumps({"restored_private_shard": index, "variant": args.variant}))
            continue
        with tempfile.TemporaryDirectory(prefix=f"objective6_{args.variant}_{name}_") as temp:
            stage = Path(temp) / name
            command = [
                sys.executable, str(ROOT / "scripts/generate_objective6_validation_shard.py"),
                "--variant", args.variant,
                "--checkpoint", str(args.checkpoint),
                "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
                "--validation-manifest", str(args.validation_manifest),
                "--validation-protocol", str(args.validation_protocol),
                "--validation-lock", str(args.validation_lock),
                "--output-dir", str(stage),
                "--shard-index", str(index), "--shard-count", str(args.shard_count),
                "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                "--image-size", str(args.image_size),
                "--maximum-length", str(args.maximum_length), "--seed", str(args.seed),
            ]
            if args.no_amp:
                command.append("--no-amp")
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, command)
            if not valid_shard(stage, args.variant, index, args.shard_count):
                raise RuntimeError(f"Generated Objective 6 shard failed verification: {name}")
            operations = [
                CommitOperationAdd(
                    path_in_repo=f"{remote_root}/{filename}",
                    path_or_fileobj=str(stage / filename),
                )
                for filename in required_names
            ]
            api.create_commit(
                repo_id=args.hf_repo, repo_type="model", token=token,
                operations=operations,
                commit_message=f"recovery: Objective 6 {args.variant} validation {name}",
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(stage, local)
            actions["generated"] += 1
            print(json.dumps({
                "private_validation_shard_uploaded": index,
                "variant": args.variant, "test_evaluated": False,
            }), flush=True)

    summaries = [
        json.loads((args.output_dir / "shards" / f"shard_{index:03d}" /
                    "shard_summary_private.json").read_text(encoding="utf-8"))
        for index in range(args.shard_count)
    ]
    total_cases = sum(int(item["cases"]) for item in summaries)
    if total_cases != 6280:
        raise RuntimeError(f"Objective 6 validation coverage changed: {total_cases}")
    inventory = {
        "artifact": "Objective 6 private validation generation inventory",
        "variant": args.variant,
        "shards": args.shard_count,
        "cases": total_cases,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "validation_protocol_sha256": VALIDATION_PROTOCOL_SHA256,
        "validation_lock_sha256": VALIDATION_LOCK_SHA256,
        "actions": actions,
        "raw_reports_public": False,
        "case_level_outputs_public": False,
        "test_manifest_opened": False,
        "test_reports_accessed": False,
        "test_evaluated": False,
        "shard_sha256": {
            f"shard_{index:03d}": summaries[index]["predictions_sha256"]
            for index in range(args.shard_count)
        },
    }
    inventory_path = args.output_dir / "private_validation_generation_inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum = inventory_path.with_suffix(".json.sha256")
    checksum.write_text(f"{sha256(inventory_path)}  {inventory_path.name}\n", encoding="utf-8")
    api.create_commit(
        repo_id=args.hf_repo, repo_type="model", token=token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{prefix}/{path.name}", path_or_fileobj=str(path)
            )
            for path in (inventory_path, checksum)
        ],
        commit_message=f"recovery: finalize Objective 6 {args.variant} validation generation",
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))
    print("OBJECTIVE 6 FULL PRIVATE VALIDATION GENERATION SUCCESSFUL")


if __name__ == "__main__":
    main()
