#!/usr/bin/env python3
"""Extract one private Objective 6 frozen-DenseNet embedding shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cxr_thesis.objective6.data import ReportGenerationDataset, collate_reports
from cxr_thesis.objective6.models import DenseNetTransformerReportGenerator
from cxr_thesis.objective6.text import ReportVocabulary


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
    parser.add_argument("--shard-index", type=int, required=True)
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


def checksum(path: Path) -> str:
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid Objective 6 embedding shard index")
    cases, manifest_hash = MANIFESTS[args.split]
    protected = {
        args.manifest: manifest_hash,
        args.checkpoint: args.expected_checkpoint_sha256,
        args.validation_protocol: PROTOCOL_SHA256,
        args.validation_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 input changed: {path}")
    frame = pd.read_csv(args.manifest, low_memory=False)
    if len(frame) != cases or set(frame["split"].astype(str)) != {args.split}:
        raise RuntimeError(f"Objective 6 {args.split} manifest changed")
    start = cases * args.shard_index // args.shard_count
    stop = cases * (args.shard_index + 1) // args.shard_count
    selected = frame.iloc[start:stop].copy().reset_index(drop=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("test_evaluated") is not False:
        raise RuntimeError("Objective 6 retrieval encoder is not test-blind")
    vocabulary = ReportVocabulary.from_dict(checkpoint["vocabulary"])
    config = checkpoint["model_config"]
    model = DenseNetTransformerReportGenerator(
        len(vocabulary.tokens), maximum_length=int(config["maximum_length"]),
        pretrained=False, freeze_image_encoder=True,
        use_clinical=bool(config["use_clinical"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    encoder = model.image_encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()
    dataset = ReportGenerationDataset(
        selected, vocabulary, image_size=args.image_size,
        maximum_length=int(config["maximum_length"]),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=collate_reports,
    )
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(
                "cuda", enabled=not args.no_amp and device.type == "cuda"
            ):
                # Match the DenseNet forward path used by Objective 6: the
                # final convolutional tensor is rectified before global pooling.
                feature_map = torch.relu(encoder(image))
                pooled = feature_map.mean(dim=(-2, -1))
            rows.append(pooled.float().cpu().numpy().astype(np.float16))
            del image, feature_map, pooled
    embeddings = np.concatenate(rows, axis=0)
    if embeddings.shape != (len(selected), 1024) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Invalid Objective 6 embedding array: {embeddings.shape}")
    args.output_dir.mkdir(parents=True)
    embedding_path = args.output_dir / "embeddings_private.npy"
    with embedding_path.open("wb") as stream:
        np.save(stream, embeddings, allow_pickle=False)
    embedding_hash = checksum(embedding_path)
    summary = {
        "artifact": "Objective 6 private retrieval embedding shard",
        "split": args.split, "shard_index": args.shard_index,
        "shard_count": args.shard_count, "start_index": start,
        "stop_index_exclusive": stop, "cases": len(selected),
        "embedding_dimension": 1024, "embedding_dtype": "float16",
        "representation": "global-average-pooled final DenseNet convolutional features",
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "manifest_sha256": manifest_hash,
        "validation_protocol_sha256": PROTOCOL_SHA256,
        "validation_lock_sha256": LOCK_SHA256,
        "embeddings_sha256": embedding_hash,
        "labels_accessed_for_retrieval": False,
        "report_content_accessed_for_retrieval": False,
        "test_manifest_opened": False, "test_evaluated": False,
        "public_upload_allowed": False,
    }
    summary_path = args.output_dir / "embedding_summary_private.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum(summary_path)
    print(json.dumps({
        "split": args.split, "shard": args.shard_index,
        "cases": len(selected), "dimension": 1024,
        "embeddings_sha256": embedding_hash,
        "test_evaluated": False, "private_only": True,
    }, sort_keys=True))
    print("OBJECTIVE 6 PRIVATE RETRIEVAL EMBEDDING SHARD SUCCESSFUL")


if __name__ == "__main__":
    main()
