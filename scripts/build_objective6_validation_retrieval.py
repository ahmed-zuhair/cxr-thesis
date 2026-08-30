#!/usr/bin/env python3
"""Build the locked Objective 6 nearest-training-image validation baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TRAIN_SHA256 = "278addf3c0a216bb206b4e4b79364f26bacbee977f3209e9275e2abbd8fda7d7"
VAL_SHA256 = "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
PROTOCOL_SHA256 = "81424c30f1619707325f0a83ef9a6fba3a859743e3b4ee0c33ac68dba6161438"
LOCK_SHA256 = "e48b11cc0af8be0866b873ae91dd5f4c55738b39927d6dec52d2f29cf5f8275a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--train-embedding-root", type=Path, required=True)
    parser.add_argument("--val-embedding-root", type=Path, required=True)
    parser.add_argument("--train-shards", type=int, default=20)
    parser.add_argument("--val-shards", type=int, default=10)
    parser.add_argument("--validation-protocol", type=Path, required=True)
    parser.add_argument("--validation-lock", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--output-shards", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embeddings(root: Path, shards: int, split: str) -> np.ndarray:
    arrays = []
    expected_start = 0
    for index in range(shards):
        directory = root / "shards" / f"shard_{index:03d}"
        array_path = directory / "embeddings_private.npy"
        checksum = array_path.with_suffix(".npy.sha256")
        summary_path = directory / "embedding_summary_private.json"
        summary_checksum = summary_path.with_suffix(".json.sha256")
        for path in (array_path, checksum, summary_path, summary_checksum):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(array_path) != checksum.read_text(encoding="utf-8").split()[0]:
            raise RuntimeError(f"Embedding checksum mismatch: {array_path}")
        if sha256(summary_path) != summary_checksum.read_text(encoding="utf-8").split()[0]:
            raise RuntimeError(f"Embedding summary checksum mismatch: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("split") != split
            or summary.get("shard_index") != index
            or summary.get("start_index") != expected_start
            or summary.get("test_evaluated") is not False
        ):
            raise RuntimeError(f"Invalid Objective 6 embedding sequence: {summary_path}")
        array = np.load(array_path, allow_pickle=False)
        if array.shape != (int(summary["cases"]), 1024):
            raise RuntimeError(f"Embedding shape mismatch: {array.shape}")
        arrays.append(array.astype(np.float32))
        expected_start = int(summary["stop_index_exclusive"])
    return np.concatenate(arrays, axis=0)


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    if args.output_dir.exists():
        raise RuntimeError("Objective 6 retrieval output already exists")
    protected = {
        args.train_manifest: TRAIN_SHA256, args.val_manifest: VAL_SHA256,
        args.validation_protocol: PROTOCOL_SHA256, args.validation_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 input changed: {path}")
    train = pd.read_csv(args.train_manifest, low_memory=False)
    validation = pd.read_csv(args.val_manifest, low_memory=False)
    if len(train) != 29283 or len(validation) != 6280:
        raise RuntimeError("Objective 6 retrieval cohort size changed")
    train_embeddings = load_embeddings(
        args.train_embedding_root, args.train_shards, "train"
    )
    val_embeddings = load_embeddings(args.val_embedding_root, args.val_shards, "val")
    if train_embeddings.shape[0] != len(train) or val_embeddings.shape[0] != len(validation):
        raise RuntimeError("Objective 6 embedding coverage mismatch")

    order = np.argsort(train["case_code"].astype(str).to_numpy(), kind="stable")
    train = train.iloc[order].reset_index(drop=True)
    train_embeddings = train_embeddings[order]
    train_norm = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    val_norm = np.linalg.norm(val_embeddings, axis=1, keepdims=True)
    if (train_norm <= 0).any() or (val_norm <= 0).any():
        raise RuntimeError("Objective 6 retrieval contains a zero embedding")
    train_embeddings /= train_norm
    val_embeddings /= val_norm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.from_numpy(train_embeddings).to(device)
    nearest_indices: list[np.ndarray] = []
    nearest_scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(validation), args.query_batch_size):
            query = torch.from_numpy(
                val_embeddings[start : start + args.query_batch_size]
            ).to(device)
            similarity = query @ train_tensor.T
            score, index = similarity.max(dim=1)
            nearest_indices.append(index.cpu().numpy())
            nearest_scores.append(score.cpu().numpy())
            del query, similarity, score, index
    neighbor = np.concatenate(nearest_indices)
    similarity = np.concatenate(nearest_scores)
    retrieved_reports = train.iloc[neighbor]["report"].fillna("").astype(str).to_numpy()
    output = pd.DataFrame({
        "case_code": validation["case_code"].astype(str),
        "patient_id": validation["patient_id"].astype(str),
        "study_id": validation["study_id"].astype(str),
        "reference_report": validation["report"].fillna("").astype(str),
        "generated_report": retrieved_reports,
        "reference_labels": validation["labels"].fillna("").astype(str),
        "nearest_cosine_similarity": similarity.astype(float),
    })
    args.output_dir.mkdir(parents=True)
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    if not bool(api.model_info(args.hf_repo, token=token).private):
        raise RuntimeError("Objective 6 retrieval recovery repository must be private")
    prefix = args.hf_path.strip("/")
    shard_hashes = {}
    upload_paths: list[tuple[Path, str]] = []
    for index in range(args.output_shards):
        start = len(output) * index // args.output_shards
        stop = len(output) * (index + 1) // args.output_shards
        directory = args.output_dir / "shards" / f"shard_{index:03d}"
        directory.mkdir(parents=True)
        path = directory / "predictions_private.csv"
        output.iloc[start:stop].to_csv(path, index=False, lineterminator="\n")
        digest = sha256(path)
        checksum = path.with_suffix(".csv.sha256")
        checksum.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
        summary = {
            "artifact": "Objective 6 private validation retrieval shard",
            "variant": "nearest_training_image_retrieval",
            "shard_index": index, "shard_count": args.output_shards,
            "start_index": start, "stop_index_exclusive": stop,
            "cases": stop - start, "predictions_sha256": digest,
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "validation_protocol_sha256": PROTOCOL_SHA256,
            "validation_lock_sha256": LOCK_SHA256,
            "labels_used_for_neighbor_selection": False,
            "report_content_used_for_neighbor_selection": False,
            "test_manifest_opened": False, "test_evaluated": False,
            "public_upload_allowed": False,
        }
        summary_path = directory / "shard_summary_private.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary_checksum = summary_path.with_suffix(".json.sha256")
        summary_checksum.write_text(
            f"{sha256(summary_path)}  {summary_path.name}\n", encoding="utf-8"
        )
        shard_hashes[f"shard_{index:03d}"] = digest
        remote_root = f"{prefix}/shards/shard_{index:03d}"
        for item in (path, checksum, summary_path, summary_checksum):
            upload_paths.append((item, f"{remote_root}/{item.name}"))
    inventory = {
        "artifact": "Objective 6 private validation retrieval inventory",
        "variant": "nearest_training_image_retrieval",
        "cases": len(output), "shards": args.output_shards,
        "training_candidates": len(train), "embedding_dimension": 1024,
        "similarity": "cosine", "tie_break": "lexicographic private case_code",
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "validation_protocol_sha256": PROTOCOL_SHA256,
        "validation_lock_sha256": LOCK_SHA256,
        "labels_used_for_neighbor_selection": False,
        "report_content_used_for_neighbor_selection": False,
        "case_level_outputs_public": False,
        "test_manifest_opened": False, "test_evaluated": False,
        "shard_sha256": shard_hashes,
    }
    inventory_path = args.output_dir / "private_validation_retrieval_inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inventory_checksum = inventory_path.with_suffix(".json.sha256")
    inventory_checksum.write_text(
        f"{sha256(inventory_path)}  {inventory_path.name}\n", encoding="utf-8"
    )
    upload_paths.extend((
        (inventory_path, f"{prefix}/{inventory_path.name}"),
        (inventory_checksum, f"{prefix}/{inventory_checksum.name}"),
    ))
    api.create_commit(
        repo_id=args.hf_repo, repo_type="model", token=token,
        operations=[
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
            for local, remote in upload_paths
        ],
        commit_message="recovery: finalize Objective 6 validation retrieval baseline",
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))
    print("OBJECTIVE 6 PRIVATE VALIDATION RETRIEVAL BASELINE SUCCESSFUL")


if __name__ == "__main__":
    main()
