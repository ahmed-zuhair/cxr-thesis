#!/usr/bin/env python3
"""Extract Objective 6 retrieval embeddings with private HF shard recovery."""

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


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "train": (29283, "278addf3c0a216bb206b4e4b79364f26bacbee977f3209e9275e2abbd8fda7d7"),
    "val": (6280, "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"),
}
PROTOCOL_SHA256 = "81424c30f1619707325f0a83ef9a6fba3a859743e3b4ee0c33ac68dba6161438"
LOCK_SHA256 = "e48b11cc0af8be0866b873ae91dd5f4c55738b39927d6dec52d2f29cf5f8275a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--validation-protocol", type=Path, required=True)
    parser.add_argument("--validation-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid(directory: Path, split: str, index: int, count: int) -> bool:
    array = directory / "embeddings_private.npy"
    array_checksum = array.with_suffix(".npy.sha256")
    summary = directory / "embedding_summary_private.json"
    summary_checksum = summary.with_suffix(".json.sha256")
    if not all(path.is_file() for path in (array, array_checksum, summary, summary_checksum)):
        return False
    if sha256(array) != array_checksum.read_text(encoding="utf-8").split()[0]:
        return False
    if sha256(summary) != summary_checksum.read_text(encoding="utf-8").split()[0]:
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return all((
        payload.get("split") == split, payload.get("shard_index") == index,
        payload.get("shard_count") == count,
        payload.get("embeddings_sha256") == sha256(array),
        payload.get("test_evaluated") is False,
        payload.get("public_upload_allowed") is False,
    ))


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    cases, manifest_hash = MANIFESTS[args.split]
    protected = {
        args.manifest: manifest_hash, args.checkpoint: args.expected_checkpoint_sha256,
        args.validation_protocol: PROTOCOL_SHA256, args.validation_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 input changed: {path}")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi(token=token)
    if not bool(api.model_info(args.hf_repo, token=token).private):
        raise RuntimeError("Objective 6 embedding recovery repository must be private")
    remote = set(api.list_repo_files(args.hf_repo, repo_type="model", token=token))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.hf_path.strip("/")
    names = (
        "embeddings_private.npy", "embeddings_private.npy.sha256",
        "embedding_summary_private.json", "embedding_summary_private.json.sha256",
    )
    actions = {"restored": 0, "generated": 0, "reused": 0}
    for index in range(args.shard_count):
        shard_name = f"shard_{index:03d}"
        local = args.output_dir / "shards" / shard_name
        if valid(local, args.split, index, args.shard_count):
            actions["reused"] += 1
            continue
        remote_root = f"{prefix}/shards/{shard_name}"
        expected_remote = {f"{remote_root}/{name}" for name in names}
        available = expected_remote & remote
        if available and available != expected_remote:
            raise RuntimeError(f"Partial remote embedding shard: {shard_name}")
        if available == expected_remote:
            local.mkdir(parents=True, exist_ok=True)
            for name in names:
                source = Path(hf_hub_download(
                    repo_id=args.hf_repo, filename=f"{remote_root}/{name}",
                    repo_type="model", token=token, force_download=True,
                ))
                shutil.copy2(source, local / name)
            if not valid(local, args.split, index, args.shard_count):
                raise RuntimeError(f"Restored embedding shard is invalid: {shard_name}")
            actions["restored"] += 1
            continue
        with tempfile.TemporaryDirectory(prefix=f"objective6_embed_{args.split}_") as temp:
            stage = Path(temp) / shard_name
            command = [
                sys.executable,
                str(ROOT / "scripts/extract_objective6_retrieval_embedding_shard.py"),
                "--split", args.split, "--manifest", str(args.manifest),
                "--checkpoint", str(args.checkpoint),
                "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
                "--validation-protocol", str(args.validation_protocol),
                "--validation-lock", str(args.validation_lock),
                "--output-dir", str(stage), "--shard-index", str(index),
                "--shard-count", str(args.shard_count),
                "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                "--image-size", str(args.image_size),
            ]
            if args.no_amp:
                command.append("--no-amp")
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command)
            if not valid(stage, args.split, index, args.shard_count):
                raise RuntimeError(f"Generated embedding shard is invalid: {shard_name}")
            api.create_commit(
                repo_id=args.hf_repo, repo_type="model", token=token,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=f"{remote_root}/{name}",
                        path_or_fileobj=str(stage / name),
                    )
                    for name in names
                ],
                commit_message=f"recovery: Objective 6 retrieval {args.split} {shard_name}",
            )
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(stage, local)
            actions["generated"] += 1
            print(json.dumps({
                "private_embedding_shard_uploaded": index,
                "split": args.split, "test_evaluated": False,
            }), flush=True)
    summaries = [
        json.loads((args.output_dir / "shards" / f"shard_{i:03d}" /
                    "embedding_summary_private.json").read_text(encoding="utf-8"))
        for i in range(args.shard_count)
    ]
    total = sum(int(item["cases"]) for item in summaries)
    if total != cases:
        raise RuntimeError(f"Objective 6 {args.split} embedding coverage changed: {total}")
    inventory = {
        "artifact": "Objective 6 private retrieval embedding inventory",
        "split": args.split, "cases": total, "shards": args.shard_count,
        "embedding_dimension": 1024, "actions": actions,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "manifest_sha256": manifest_hash,
        "validation_protocol_sha256": PROTOCOL_SHA256,
        "validation_lock_sha256": LOCK_SHA256,
        "labels_accessed_for_retrieval": False,
        "report_content_accessed_for_retrieval": False,
        "test_manifest_opened": False, "test_evaluated": False,
        "public_upload_allowed": False,
        "shard_sha256": {
            f"shard_{i:03d}": summaries[i]["embeddings_sha256"]
            for i in range(args.shard_count)
        },
    }
    inventory_path = args.output_dir / "private_embedding_inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_path = inventory_path.with_suffix(".json.sha256")
    checksum_path.write_text(
        f"{sha256(inventory_path)}  {inventory_path.name}\n", encoding="utf-8"
    )
    api.create_commit(
        repo_id=args.hf_repo, repo_type="model", token=token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{prefix}/{path.name}", path_or_fileobj=str(path)
            )
            for path in (inventory_path, checksum_path)
        ],
        commit_message=f"recovery: finalize Objective 6 retrieval {args.split} embeddings",
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))
    print("OBJECTIVE 6 FULL PRIVATE RETRIEVAL EMBEDDINGS SUCCESSFUL")


if __name__ == "__main__":
    main()
