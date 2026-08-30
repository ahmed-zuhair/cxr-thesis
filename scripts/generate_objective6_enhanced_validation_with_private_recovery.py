#!/usr/bin/env python3
"""Generate Objective 6 v1.1 validation shards with private HF recovery."""

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

VARIANT = "clinical_guided_multimodal_v1_1"
VALIDATION_MANIFEST_SHA256 = (
    "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
)
ENHANCEMENT_PROTOCOL_SHA256 = (
    "279e4fe83da6d82afcbcce595b5596980ca970fae958a16264d3b3e5172eb1a1"
)
ENHANCEMENT_LOCK_SHA256 = (
    "b840440da16023c0169eb3f32c0f4ce7a20ecfa34f8f6b6bfa8ef20511aa53e6"
)
CHECKPOINT_SHA256 = (
    "bc6c6c27208b31a597890b2f12abc35a7e0979b80d6b77cd5b2e341d43baf89b"
)
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--enhancement-protocol", type=Path, required=True)
    parser.add_argument("--enhancement-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--shard-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
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


def valid_shard(directory: Path, index: int, count: int) -> bool:
    predictions = directory / "predictions_private.csv"
    checksum = predictions.with_suffix(".csv.sha256")
    summary_path = directory / "shard_summary_private.json"
    summary_checksum = summary_path.with_suffix(".json.sha256")
    required = (predictions, checksum, summary_path, summary_checksum)
    if not all(path.is_file() for path in required):
        return False
    if sha256(predictions) != checksum.read_text(encoding="utf-8").split()[0]:
        return False
    if sha256(summary_path) != summary_checksum.read_text(
        encoding="utf-8"
    ).split()[0]:
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return all((
        summary.get("variant") == VARIANT,
        summary.get("shard_index") == index,
        summary.get("shard_count") == count,
        summary.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        summary.get("predictions_sha256") == sha256(predictions),
        summary.get("validation_manifest_sha256")
        == VALIDATION_MANIFEST_SHA256,
        summary.get("enhancement_protocol_sha256")
        == ENHANCEMENT_PROTOCOL_SHA256,
        summary.get("enhancement_lock_sha256") == ENHANCEMENT_LOCK_SHA256,
        summary.get("test_evaluated") is False,
        summary.get("public_upload_allowed") is False,
    ))


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    if args.expected_checkpoint_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("Objective 6 v1.1 checkpoint selection changed")
    if args.shard_count != 20:
        raise RuntimeError("Objective 6 v1.1 shard count changed")
    protected = {
        args.validation_manifest: VALIDATION_MANIFEST_SHA256,
        args.enhancement_protocol: ENHANCEMENT_PROTOCOL_SHA256,
        args.enhancement_lock: ENHANCEMENT_LOCK_SHA256,
        args.checkpoint: CHECKPOINT_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 v1.1 input changed: {path}")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    if not bool(api.model_info(args.hf_repo, token=token).private):
        raise RuntimeError("Objective 6 v1.1 recovery repository must be private")
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
        if valid_shard(local, index, args.shard_count):
            actions["reused"] += 1
            continue
        remote_root = f"{prefix}/shards/{name}"
        remote_names = {f"{remote_root}/{item}" for item in required_names}
        available = remote_names & remote
        if available and available != remote_names:
            raise RuntimeError(f"Partial remote Objective 6 v1.1 shard: {name}")
        if available == remote_names:
            local.mkdir(parents=True, exist_ok=True)
            for filename in required_names:
                downloaded = Path(hf_hub_download(
                    repo_id=args.hf_repo, filename=f"{remote_root}/{filename}",
                    repo_type="model", token=token, force_download=True,
                ))
                shutil.copy2(downloaded, local / filename)
            if not valid_shard(local, index, args.shard_count):
                raise RuntimeError(f"Remote Objective 6 v1.1 shard invalid: {name}")
            actions["restored"] += 1
            print(json.dumps({"restored_private_shard": index, "variant": VARIANT}))
            continue
        with tempfile.TemporaryDirectory(prefix=f"objective6_v1_1_{name}_") as temp:
            stage = Path(temp) / name
            command = [
                sys.executable,
                str(ROOT / "scripts/generate_objective6_enhanced_validation_shard.py"),
                "--checkpoint", str(args.checkpoint),
                "--expected-checkpoint-sha256", CHECKPOINT_SHA256,
                "--validation-manifest", str(args.validation_manifest),
                "--enhancement-protocol", str(args.enhancement_protocol),
                "--enhancement-lock", str(args.enhancement_lock),
                "--output-dir", str(stage),
                "--shard-index", str(index),
                "--shard-count", str(args.shard_count),
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
                "--image-size", str(args.image_size),
                "--maximum-length", str(args.maximum_length),
                "--seed", str(args.seed),
            ]
            if args.no_amp:
                command.append("--no-amp")
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, command)
            if not valid_shard(stage, index, args.shard_count):
                raise RuntimeError(f"Generated Objective 6 v1.1 shard invalid: {name}")
            api.create_commit(
                repo_id=args.hf_repo, repo_type="model", token=token,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=f"{remote_root}/{filename}",
                        path_or_fileobj=str(stage / filename),
                    )
                    for filename in required_names
                ],
                commit_message=f"recovery: Objective 6 v1.1 validation {name}",
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(stage, local)
            actions["generated"] += 1
            print(json.dumps({
                "private_validation_shard_uploaded": index,
                "variant": VARIANT, "test_evaluated": False,
            }), flush=True)

    summaries = [
        json.loads((
            args.output_dir / "shards" / f"shard_{index:03d}"
            / "shard_summary_private.json"
        ).read_text(encoding="utf-8"))
        for index in range(args.shard_count)
    ]
    total_cases = sum(int(item["cases"]) for item in summaries)
    if total_cases != 6280:
        raise RuntimeError(f"Objective 6 v1.1 coverage changed: {total_cases}")
    inventory = {
        "artifact": "Objective 6 v1.1 private validation generation inventory",
        "variant": VARIANT, "shards": args.shard_count, "cases": total_cases,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "enhancement_protocol_sha256": ENHANCEMENT_PROTOCOL_SHA256,
        "enhancement_lock_sha256": ENHANCEMENT_LOCK_SHA256,
        "decoding": {
            "method": "deterministic beam search", "beam_width": 3,
            "length_normalization_alpha": 0.7, "no_repeat_ngram_size": 4,
        },
        "actions": actions, "raw_reports_public": False,
        "case_level_outputs_public": False, "test_manifest_opened": False,
        "test_reports_accessed": False, "test_evaluated": False,
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
    checksum.write_text(
        f"{sha256(inventory_path)}  {inventory_path.name}\n", encoding="utf-8"
    )
    api.create_commit(
        repo_id=args.hf_repo, repo_type="model", token=token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{prefix}/{path.name}", path_or_fileobj=str(path)
            )
            for path in (inventory_path, checksum)
        ],
        commit_message="recovery: finalize Objective 6 v1.1 validation generation",
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))
    print("OBJECTIVE 6 V1.1 FULL PRIVATE VALIDATION GENERATION SUCCESSFUL")


if __name__ == "__main__":
    main()
