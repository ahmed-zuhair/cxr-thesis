"""Generate compact ROI patch graphs for Objective 2 GCN/GAT training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.config import load_config
from cxr_thesis.objective1.graphs import GraphSample
from cxr_thesis.objective1.manifest import validate_manifest
from cxr_thesis.object1.preprocessing import load_image, preprocess_cxr
from cxr_thesis.object1.segmentation import UNet2D
from cxr_thesis.objective2.graph_generation import (
    build_frozen_roi_graph,
    safe_graph_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen-model ROI graphs without saving predicted masks"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "objective1" / "default.yaml",
    )
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-component-fraction", type=float, default=0.001)
    parser.add_argument("--min-component-pixels", type=int, default=0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(value: object, root: Path) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else root / candidate


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("Batch size and save interval must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Limit must be positive")

    manifest_hash = sha256_file(args.manifest)
    checkpoint_hash = sha256_file(args.checkpoint)
    if args.expected_manifest_sha256 and manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError("Graph-generation manifest SHA-256 does not match")
    if args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256:
        raise RuntimeError("Segmentation checkpoint SHA-256 does not match")

    frame = pd.read_csv(
        args.manifest,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    validate_manifest(frame, require_files=True, root=args.data_root)
    if args.limit is not None:
        frame = frame.iloc[: args.limit].copy()
    records = frame.to_dict(orient="records")

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "UNet2D":
        raise ValueError("The graph generator requires a frozen UNet2D checkpoint")
    channels = tuple(int(value) for value in checkpoint.get("channels", (32, 64, 128, 256)))
    model = UNet2D(channels=channels)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(
        checkpoint.get("validation_metrics", {}).get(
            "threshold", config.segmentation.threshold
        )
    )
    device = select_device(args.device)
    model = model.to(device).eval()

    args.graph_dir.mkdir(parents=True, exist_ok=True)
    audit_by_id: dict[str, dict[str, object]] = {}
    if args.resume and args.audit_csv.is_file():
        previous = pd.read_csv(args.audit_csv, dtype={"image_id": str})
        audit_by_id = {
            str(row["image_id"]): row.to_dict()
            for _, row in previous.iterrows()
        }

    pending: list[dict[str, object]] = []
    resumed = 0
    for record in records:
        image_id = str(record["image_id"])
        target = args.graph_dir / f"{safe_graph_name(image_id)}.npz"
        prior = audit_by_id.get(image_id, {})
        compatible = (
            args.resume
            and target.is_file()
            and prior.get("status") == "complete"
            and str(prior.get("checkpoint_sha256")) == checkpoint_hash
            and str(prior.get("manifest_sha256")) == manifest_hash
        )
        if compatible:
            GraphSample.load(target)
            resumed += 1
        else:
            if target.exists():
                raise FileExistsError(f"Non-resumable graph already exists: {target}")
            pending.append(record)

    print(
        json.dumps(
            {
                "event": "start",
                "manifest_rows": len(records),
                "pending": len(pending),
                "resumed": resumed,
                "batch_size": args.batch_size,
                "device": str(device),
                "manifest_sha256": manifest_hash,
                "checkpoint_sha256": checkpoint_hash,
                "threshold": threshold,
                "predicted_masks_saved": False,
            }
        )
    )

    generated = 0
    failed = 0
    started = time.perf_counter()
    last_saved = 0
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start : batch_start + args.batch_size]
        loaded: list[tuple[dict[str, object], np.ndarray]] = []
        for record in batch:
            image_id = str(record["image_id"])
            try:
                image = load_image(resolve(record["image_path"], args.data_root))
                processed, _ = preprocess_cxr(image, config.preprocessing)
                loaded.append((record, processed))
            except Exception as error:
                failed += 1
                audit_by_id[image_id] = {
                    "image_id": image_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "manifest_sha256": manifest_hash,
                    "checkpoint_sha256": checkpoint_hash,
                }
                if not args.continue_on_error:
                    raise

        if loaded:
            tensor = torch.from_numpy(
                np.stack([item[1] for item in loaded]).astype(np.float32) / 255.0
            )[:, None].to(device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(tensor)
                probabilities = torch.sigmoid(logits)[:, 0].float().cpu().numpy()

            for position, (record, processed) in enumerate(loaded):
                image_id = str(record["image_id"])
                target = args.graph_dir / f"{safe_graph_name(image_id)}.npz"
                try:
                    result = build_frozen_roi_graph(
                        processed,
                        probabilities[position],
                        threshold=threshold,
                        config=config,
                        record=record,
                        checkpoint_sha256=checkpoint_hash,
                        min_component_fraction=args.min_component_fraction,
                        min_component_pixels=args.min_component_pixels,
                    )
                    result.graph.save(target)
                    audit_by_id[image_id] = {
                        "image_id": image_id,
                        "status": "complete",
                        "split": str(record["split"]),
                        "graph_nodes": int(result.graph.x.shape[0]),
                        "graph_edges": int(result.graph.edge_index.shape[1]),
                        "node_features": int(result.graph.x.shape[1]),
                        **result.mask_quality,
                        "components_before_cleanup": int(result.cleanup["components_before"]),
                        "components_after_cleanup": int(result.cleanup["components_after"]),
                        "manifest_sha256": manifest_hash,
                        "checkpoint_sha256": checkpoint_hash,
                        "graph_path": str(target),
                        "error": "",
                    }
                    generated += 1
                except Exception as error:
                    failed += 1
                    audit_by_id[image_id] = {
                        "image_id": image_id,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "manifest_sha256": manifest_hash,
                        "checkpoint_sha256": checkpoint_hash,
                    }
                    if not args.continue_on_error:
                        raise

        completed = generated + failed
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "event": "progress",
                    "completed": resumed + completed,
                    "total": len(records),
                    "generated": generated,
                    "resumed": resumed,
                    "failed": failed,
                    "elapsed_seconds": elapsed,
                }
            )
        )
        if completed - last_saved >= args.save_every:
            atomic_table(pd.DataFrame(audit_by_id.values()), args.audit_csv)
            last_saved = completed

    audit = pd.DataFrame(audit_by_id.values())
    atomic_table(audit, args.audit_csv)
    complete = int((audit["status"] == "complete").sum()) if not audit.empty else 0
    failures = int((audit["status"] == "failed").sum()) if not audit.empty else 0
    summary = {
        "artifact": "Objective 2 frozen-ROI patch graph generation",
        "manifest_rows": len(records),
        "complete_graphs": complete,
        "failed_graphs": failures,
        "resumed_graphs": resumed,
        "node_feature_dimension": 7,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "threshold": threshold,
        "predicted_masks_saved": False,
        "original_medical_images_copied": False,
        "test_evaluated": False,
        "audit_csv_sha256": sha256_file(args.audit_csv),
    }
    atomic_json(summary, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"Graph generation completed with {failures} failed cases")
    print("OBJECTIVE 2 ROI GRAPH GENERATION SUCCESSFUL")


if __name__ == "__main__":
    main()
