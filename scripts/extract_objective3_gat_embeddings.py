#!/usr/bin/env python3
"""Extract frozen GAT embeddings with private, shard-level HF recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective2.data import (
    GraphClassificationDataset,
    collate_graph_samples,
)
from cxr_thesis.objective2.graph_generation import safe_graph_name
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective2.training import seed_everything
from cxr_thesis.objective3.embeddings import (
    load_embedding_shard,
    save_embedding_shard,
)

PRIMARY_LABELS = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Consolidation",
    "Pneumothorax",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Emphysema",
    "Edema",
    "Fibrosis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-train-cases", type=int, default=30_000)
    parser.add_argument("--expected-val-cases", type=int, default=5_000)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_manifest(
    path: Path,
    split: str,
    expected_hash: str,
    expected_cases: int,
) -> pd.DataFrame:
    if sha256_file(path) != expected_hash:
        raise RuntimeError(f"{split} manifest SHA-256 does not match")
    frame = pd.read_csv(
        path,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    if len(frame) != expected_cases:
        raise RuntimeError(f"{split} case count does not match")
    if set(frame["split"].astype(str).str.lower()) != {split}:
        raise RuntimeError(f"{split} manifest contains another split")
    if frame["image_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{split} manifest contains duplicate image IDs")
    missing_labels = sorted(
        {f"label_{label}" for label in PRIMARY_LABELS} - set(frame.columns)
    )
    if missing_labels:
        raise RuntimeError(f"{split} labels are missing: {missing_labels}")
    return frame


def validate_checkpoint(path: Path, expected_hash: str) -> dict[str, object]:
    if sha256_file(path) != expected_hash:
        raise RuntimeError("Frozen GAT checkpoint SHA-256 does not match")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "model": checkpoint.get("model_name") == "gat",
        "labels": checkpoint.get("label_names") == PRIMARY_LABELS,
        "test_blind": checkpoint.get("test_evaluated") is False,
        "state": isinstance(checkpoint.get("model_state"), dict),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen GAT checkpoint validation failed: {checks}")
    return checkpoint


def shard_valid(path: Path, checksum: Path, image_ids: list[str]) -> bool:
    if not path.is_file() or not checksum.is_file():
        return False
    recorded = checksum.read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != recorded:
        return False
    try:
        load_embedding_shard(path, expected_image_ids=image_ids)
    except (OSError, ValueError):
        return False
    return True


@torch.inference_mode()
def extract_shard(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    graph_root: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    labels = [f"label_{label}" for label in PRIMARY_LABELS]
    dataset = GraphClassificationDataset(frame, labels, graph_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_graph_samples,
    )
    outputs: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        embedding = model.encode(batch.to(device))
        outputs.append(embedding.float().cpu().numpy())
    result = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if result.shape != (len(frame), 160) or not np.isfinite(result).all():
        raise RuntimeError("Frozen GAT produced invalid embeddings")
    return result


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError(
            "Shard size and batch size must be positive; workers non-negative"
        )
    if args.expected_train_cases % args.shard_size:
        raise ValueError("Shard size must preserve the train/validation boundary")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    train = read_manifest(
        args.train_manifest,
        "train",
        args.expected_train_sha256,
        args.expected_train_cases,
    )
    validation = read_manifest(
        args.val_manifest,
        "val",
        args.expected_val_sha256,
        args.expected_val_cases,
    )
    if set(train["patient_id"]) & set(validation["patient_id"]):
        raise RuntimeError("Patient leakage exists between training and validation")
    combined = pd.concat([train, validation], ignore_index=True)
    expected_graphs = {
        f"{safe_graph_name(value)}.npz" for value in combined["image_id"].astype(str)
    }
    actual_graphs = {path.name for path in args.graph_root.glob("*.npz")}
    if actual_graphs != expected_graphs:
        raise RuntimeError(
            f"Graph root mismatch: expected {len(expected_graphs)}, "
            f"found {len(actual_graphs)}"
        )
    checkpoint = validate_checkpoint(args.checkpoint, args.expected_checkpoint_sha256)
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model = build_classifier("gat", len(PRIMARY_LABELS), node_dim=7, clinical_dim=9)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.requires_grad_(False).to(device).eval()

    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub before extraction") from error
    api = HfApi(token=token)
    info = api.model_info(args.hf_repo, token=token)
    if not bool(info.private):
        raise RuntimeError("Embedding recovery repository must remain private")
    remote_files = set(
        api.list_repo_files(args.hf_repo, repo_type="model", token=token)
    )
    prefix = args.hf_path.strip("/")
    shard_root = args.output_root / "private" / "shards"
    download_root = args.output_root / ".downloads"
    shard_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)
    total_shards = (len(combined) + args.shard_size - 1) // args.shard_size
    records: list[dict[str, object]] = []

    print(
        json.dumps(
            {
                "event": "objective3_embedding_extraction_start",
                "train_cases": len(train),
                "validation_cases": len(validation),
                "total_shards": total_shards,
                "checkpoint_sha256": args.expected_checkpoint_sha256,
                "encoder": "frozen_gat",
                "embedding_dimension": 160,
                "device": str(device),
                "test_manifest_opened": False,
                "test_labels_accessed": False,
            }
        )
    )

    for shard_index, start in enumerate(range(0, len(combined), args.shard_size)):
        stop = min(start + args.shard_size, len(combined))
        shard_name = f"shard-{shard_index:05d}"
        frame = combined.iloc[start:stop].copy()
        image_ids = frame["image_id"].astype(str).tolist()
        local_shard = shard_root / f"{shard_name}.npz"
        local_checksum = shard_root / f"{shard_name}.npz.sha256"
        remote_shard = f"{prefix}/shards/{shard_name}.npz"
        remote_checksum = f"{remote_shard}.sha256"
        has_remote_shard = remote_shard in remote_files
        has_remote_checksum = remote_checksum in remote_files
        if has_remote_shard != has_remote_checksum:
            raise RuntimeError(f"Incomplete remote recovery pair for {shard_name}")
        remote_expected_hash: str | None = None
        downloaded_checksum: Path | None = None
        if has_remote_checksum:
            downloaded_checksum = Path(
                hf_hub_download(
                    args.hf_repo,
                    filename=remote_checksum,
                    repo_type="model",
                    token=token,
                    local_dir=download_root,
                    force_download=True,
                )
            )
            remote_expected_hash = downloaded_checksum.read_text(
                encoding="utf-8"
            ).split()[0]
        action = "local"
        local_is_valid = shard_valid(local_shard, local_checksum, image_ids)
        local_matches_remote = remote_expected_hash is None or (
            local_is_valid and sha256_file(local_shard) == remote_expected_hash
        )
        if not local_is_valid or not local_matches_remote:
            local_shard.unlink(missing_ok=True)
            local_checksum.unlink(missing_ok=True)
            if has_remote_shard:
                downloaded_shard = Path(
                    hf_hub_download(
                        args.hf_repo,
                        filename=remote_shard,
                        repo_type="model",
                        token=token,
                        local_dir=download_root,
                        force_download=True,
                    )
                )
                if downloaded_checksum is None:
                    raise RuntimeError(f"Missing remote checksum for {shard_name}")
                shutil.copy2(downloaded_shard, local_shard)
                shutil.copy2(downloaded_checksum, local_checksum)
                if not shard_valid(local_shard, local_checksum, image_ids):
                    raise RuntimeError(
                        f"Restored shard failed validation: {shard_name}"
                    )
                action = "restored"
            else:
                embeddings = extract_shard(
                    model,
                    frame,
                    args.graph_root,
                    device,
                    args.batch_size,
                    args.workers,
                )
                save_embedding_shard(local_shard, embeddings, image_ids)
                digest = sha256_file(local_shard)
                local_checksum.write_text(
                    f"{digest}  {local_shard.name}\n", encoding="utf-8"
                )
                action = "generated"
        digest = sha256_file(local_shard)
        embeddings, _ = load_embedding_shard(local_shard, expected_image_ids=image_ids)
        if has_remote_shard:
            if digest != remote_expected_hash:
                raise RuntimeError(f"Remote checksum mismatch for {shard_name}")
        else:
            api.create_commit(
                repo_id=args.hf_repo,
                repo_type="model",
                token=token,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=remote_shard,
                        path_or_fileobj=str(local_shard),
                    ),
                    CommitOperationAdd(
                        path_in_repo=remote_checksum,
                        path_or_fileobj=str(local_checksum),
                    ),
                ],
                commit_message=f"recovery: add Objective 3 embedding {shard_name}",
            )
            remote_files.update({remote_shard, remote_checksum})
        records.append(
            {
                "shard": shard_name,
                "start": start,
                "stop": stop,
                "cases": len(frame),
                "split": str(frame["split"].iloc[0]).lower(),
                "sha256": digest,
                "embedding_mean": float(embeddings.mean()),
                "embedding_standard_deviation": float(embeddings.std()),
                "action": action,
            }
        )
        print(
            json.dumps(
                {
                    "event": "objective3_embedding_shard_complete",
                    "index": shard_index + 1,
                    "total": total_shards,
                    "shard": shard_name,
                    "cases": len(frame),
                    "action": action,
                }
            ),
            flush=True,
        )

    index = {
        "artifact": "Private Objective 3 frozen GAT embeddings",
        "encoder": "gat",
        "encoder_frozen": True,
        "embedding_dimension": 160,
        "train_manifest_sha256": args.expected_train_sha256,
        "validation_manifest_sha256": args.expected_val_sha256,
        "gat_checkpoint_sha256": args.expected_checkpoint_sha256,
        "train_cases": len(train),
        "validation_cases": len(validation),
        "complete_embeddings": len(combined),
        "shard_size": args.shard_size,
        "shards": records,
        "test_manifest_opened": False,
        "test_labels_accessed": False,
        "test_evaluated": False,
        "medical_images_copied": False,
        "predicted_masks_saved": False,
        "case_identifiers_private": True,
        "allowed_for_public_upload": False,
    }
    index_path = args.output_root / "private" / "embedding_recovery_index.json"
    atomic_json(index, index_path)
    api.upload_file(
        path_or_fileobj=str(index_path),
        path_in_repo=f"{prefix}/embedding_recovery_index.json",
        repo_id=args.hf_repo,
        repo_type="model",
        token=token,
        commit_message="recovery: finalize Objective 3 frozen GAT embeddings",
    )
    print(json.dumps(index, indent=2, sort_keys=True))
    print("OBJECTIVE 3 FROZEN GAT EMBEDDING EXTRACTION SUCCESSFUL")


if __name__ == "__main__":
    main()
