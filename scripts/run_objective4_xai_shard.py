#!/usr/bin/env python3
"""Compute one privately recoverable Objective 4 quantitative-XAI shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.config import load_config
from cxr_thesis.objective1.preprocessing import (
    load_image,
    preprocess_cxr,
    restore_mask,
)
from cxr_thesis.objective1.segmentation import UNet2D, remove_small_components
from cxr_thesis.objective2.data import ImageClassificationDataset
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective2.training import seed_everything
from cxr_thesis.objective4 import (
    GradCAM,
    deletion_insertion_auc,
    imagenet_gamma_perturbation,
    integrated_gradients,
    saliency_concentration,
    saliency_spearman,
)


PRIMARY_LABELS = [
    "Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass",
    "Consolidation", "Pneumothorax", "Pleural_Thickening",
    "Cardiomegaly", "Emphysema", "Edema", "Fibrosis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--expected-cohort-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classifier-sha256", required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-segmentation-sha256", required=True)
    parser.add_argument("--segmentation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=12)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--ig-internal-batch-size", type=int, default=8)
    parser.add_argument("--faithfulness-steps", type=int, default=11)
    parser.add_argument("--stability-gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_classifier(path: Path, expected_hash: str, device: torch.device):
    if sha256_file(path) != expected_hash:
        raise RuntimeError("DenseNet checkpoint SHA-256 does not match")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "densenet121":
        raise RuntimeError("Objective 4 classifier is not DenseNet-121")
    if checkpoint.get("label_names") != PRIMARY_LABELS:
        raise RuntimeError("DenseNet label order does not match Objective 4")
    if checkpoint.get("test_evaluated") is not False:
        raise RuntimeError("DenseNet checkpoint is not test-blind")
    configuration = dict(checkpoint.get("model_config") or {})
    signature = dict(checkpoint.get("training_signature") or {})
    if int(configuration.get("image_size", signature.get("image_size", 0))) != 320:
        raise RuntimeError("DenseNet input size is not the frozen 320 pixels")
    dropout = float(configuration.get("dropout", signature.get("dropout", 0.2)))
    model = build_classifier(
        "densenet121", len(PRIMARY_LABELS), image_size=320,
        pretrained=False, dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def load_segmenter(path: Path, expected_hash: str, device: torch.device):
    if sha256_file(path) != expected_hash:
        raise RuntimeError("Adapted ROI checkpoint SHA-256 does not match")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "UNet2D":
        raise RuntimeError("Adapted ROI checkpoint architecture is invalid")
    channels = tuple(int(value) for value in checkpoint.get("channels", (32, 64, 128, 256)))
    model = UNet2D(channels=channels)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint.get("validation_metrics", {}).get("threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise RuntimeError("Adapted ROI threshold is invalid")
    return model.to(device).eval(), threshold


def lung_mask_for_record(
    record: pd.Series,
    *,
    root: Path,
    segmenter,
    threshold: float,
    segmentation_config,
    device: torch.device,
) -> torch.Tensor:
    source = load_image(resolve(record["image_path"], root))
    if source.ndim == 3:
        source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    processed, geometry = preprocess_cxr(source, segmentation_config.preprocessing)
    tensor = torch.from_numpy(processed.astype(np.float32) / 255.0)[None, None].to(device)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = segmenter(tensor)
        probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    cleaned, _ = remove_small_components(
        probability >= threshold, min_component_fraction=0.001,
        min_component_pixels=0,
    )
    restored = restore_mask(cleaned, geometry)
    resized = cv2.resize(restored, (320, 320), interpolation=cv2.INTER_NEAREST)
    if not np.any(resized):
        raise RuntimeError("Adapted ROI model produced an empty mask")
    return torch.from_numpy(resized.astype(bool)).to(device)


def explain(model, layer, image, clinical, target, *, ig_steps, ig_batch):
    with GradCAM(model, layer) as explainer:
        grad_cam, logits = explainer(image, clinical, target)
    integrated, integrated_logits = integrated_gradients(
        model, image, clinical, target,
        steps=ig_steps, internal_batch_size=ig_batch,
    )
    if float((logits - integrated_logits).abs().max()) > 1e-5:
        raise RuntimeError("Explanation methods produced inconsistent logits")
    return grad_cam, integrated, logits


def main() -> None:
    args = parse_args()
    if min(args.shard_size, args.ig_steps, args.ig_internal_batch_size, args.faithfulness_steps) <= 0:
        raise ValueError("Shard and metric sizes must be positive")
    if args.shard_index < 0:
        raise ValueError("Shard index cannot be negative")
    if sha256_file(args.cohort) != args.expected_cohort_sha256:
        raise RuntimeError("Objective 4 private cohort SHA-256 does not match")

    frame = pd.read_csv(args.cohort)
    required = {
        "patient_id", "image_id", "image_path", "split",
        "xai_target_label", "xai_target_index", "xai_selection_rank",
        *[f"label_{label}" for label in PRIMARY_LABELS],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Objective 4 cohort columns are missing: {missing}")
    frame = frame.sort_values(
        ["xai_target_index", "xai_selection_rank"], kind="stable"
    ).reset_index(drop=True)
    if len(frame) != 240 or frame["patient_id"].astype(str).nunique() != 240:
        raise RuntimeError("Objective 4 cohort identity is invalid")
    if set(frame["split"].astype(str).str.lower()) != {"val"}:
        raise RuntimeError("Objective 4 full XAI accepts validation rows only")
    expected_indices = frame["xai_target_label"].map(
        {label: index for index, label in enumerate(PRIMARY_LABELS)}
    )
    if expected_indices.isna().any() or not np.array_equal(
        expected_indices.to_numpy(dtype=np.int64),
        frame["xai_target_index"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Objective 4 target label/index mapping is invalid")
    target_positive = np.asarray([
        frame.iloc[index][f"label_{label}"]
        for index, label in enumerate(frame["xai_target_label"].astype(str))
    ])
    if not np.all(target_positive == 1):
        raise RuntimeError("Objective 4 cohort contains a non-positive target label")
    if set(frame["xai_target_label"].value_counts().to_list()) != {20}:
        raise RuntimeError("Objective 4 cohort is not balanced at 20 cases per label")
    start = args.shard_index * args.shard_size
    stop = min(len(frame), start + args.shard_size)
    if start >= len(frame):
        raise ValueError("Shard index is outside the Objective 4 cohort")
    shard = frame.iloc[start:stop].reset_index(drop=True)
    shard_root = args.output_dir / "private/shards" / f"shard_{args.shard_index:03d}"
    if shard_root.exists():
        raise FileExistsError(f"Objective 4 shard already exists: {shard_root}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = load_classifier(
        args.classifier_checkpoint, args.expected_classifier_sha256, device
    )
    segmenter, segmentation_threshold = load_segmenter(
        args.segmentation_checkpoint, args.expected_segmentation_sha256, device
    )
    segmentation_config = load_config(args.segmentation_config)
    dataset = ImageClassificationDataset(
        shard, [f"label_{label}" for label in PRIMARY_LABELS],
        data_root=args.data_root, image_size=320, augment=False, seed=args.seed,
        output_channels=3, normalisation="imagenet",
    )
    layer = classifier.encoder.features.norm5
    metrics: list[dict[str, object]] = []
    grad_cam_maps: list[np.ndarray] = []
    integrated_maps: list[np.ndarray] = []
    started = time.perf_counter()

    for local_index in range(len(dataset)):
        record = shard.iloc[local_index]
        sample = dataset[local_index]
        image = sample["image"].unsqueeze(0).to(device)
        clinical = sample["clinical"].unsqueeze(0).to(device)
        target = int(record["xai_target_index"])
        target_label = str(record["xai_target_label"])
        lung_mask = lung_mask_for_record(
            record, root=args.data_root, segmenter=segmenter,
            threshold=segmentation_threshold,
            segmentation_config=segmentation_config, device=device,
        )
        grad_cam, integrated, logits = explain(
            classifier, layer, image, clinical, target,
            ig_steps=args.ig_steps, ig_batch=args.ig_internal_batch_size,
        )
        perturbed = imagenet_gamma_perturbation(image, args.stability_gamma)
        perturbed_cam, perturbed_ig, _ = explain(
            classifier, layer, perturbed, clinical, target,
            ig_steps=args.ig_steps, ig_batch=args.ig_internal_batch_size,
        )
        agreement = saliency_spearman(grad_cam, integrated)
        common = {
            "patient_id": str(record["patient_id"]),
            "image_id": str(record["image_id"]),
            "target_label": target_label,
            "target_index": target,
            "target_probability": float(torch.sigmoid(logits[0, target]).item()),
            "lung_roi_fraction": float(lung_mask.float().mean().item()),
        }
        for method, saliency, stable_saliency in (
            ("grad_cam", grad_cam, perturbed_cam),
            ("integrated_gradients", integrated, perturbed_ig),
        ):
            faithfulness = deletion_insertion_auc(
                classifier, image, clinical, target, saliency,
                steps=args.faithfulness_steps,
            )
            metrics.append({
                **common,
                "method": method,
                "deletion_auc": faithfulness["deletion_auc"],
                "insertion_auc": faithfulness["insertion_auc"],
                "stability_spearman": saliency_spearman(saliency, stable_saliency),
                "lung_roi_concentration": saliency_concentration(saliency, lung_mask),
                "method_agreement_spearman": agreement,
            })
        grad_cam_maps.append(grad_cam[0, 0].detach().cpu().numpy().astype(np.float16))
        integrated_maps.append(integrated[0, 0].detach().cpu().numpy().astype(np.float16))
        print(
            f"Shard {args.shard_index:03d}: processed {local_index + 1}/{len(dataset)}; "
            f"target={target_label}"
        )
        del image, clinical, lung_mask, grad_cam, integrated, perturbed
        del perturbed_cam, perturbed_ig, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_path = shard_root / f"shard_{args.shard_index:03d}_metrics_private.csv"
    maps_path = shard_root / f"shard_{args.shard_index:03d}_saliency_private.npz"
    summary_path = shard_root / f"shard_{args.shard_index:03d}_summary_private.json"
    atomic_csv(pd.DataFrame(metrics), metrics_path)
    atomic_npz(
        maps_path,
        grad_cam=np.stack(grad_cam_maps),
        integrated_gradients=np.stack(integrated_maps),
        image_id=shard["image_id"].astype(str).to_numpy(dtype=str),
        target_index=shard["xai_target_index"].to_numpy(dtype=np.int16),
    )
    summary = {
        "artifact": "Objective 4 private quantitative-XAI shard",
        "shard_index": args.shard_index,
        "start_index": start,
        "stop_index_exclusive": stop,
        "cases": len(shard),
        "metric_rows": len(metrics),
        "cohort_sha256": args.expected_cohort_sha256,
        "classifier_sha256": args.expected_classifier_sha256,
        "segmentation_sha256": args.expected_segmentation_sha256,
        "segmentation_threshold": segmentation_threshold,
        "integrated_gradients_steps": args.ig_steps,
        "integrated_gradients_internal_batch_size": args.ig_internal_batch_size,
        "faithfulness_steps": args.faithfulness_steps,
        "stability_gamma": args.stability_gamma,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics_sha256": sha256_file(metrics_path),
        "saliency_sha256": sha256_file(maps_path),
        "test_manifest_opened": False,
        "test_labels_accessed": False,
        "test_evaluated": False,
        "medical_images_saved": False,
        "predicted_masks_saved": False,
        "saliency_maps_private": True,
        "allowed_for_public_upload": False,
    }
    atomic_json(summary, summary_path)
    (summary_path.with_suffix(".sha256")).write_text(
        f"{sha256_file(summary_path)}  {summary_path.name}\n", encoding="utf-8"
    )
    print("--- OBJECTIVE 4 XAI SHARD COMPLETE ---")
    print("Shard index:", args.shard_index)
    print("Cases:", len(shard))
    print("Metric rows:", len(metrics))
    print("Metrics SHA-256:", summary["metrics_sha256"])
    print("Saliency SHA-256:", summary["saliency_sha256"])
    print("Test manifest opened:", False)
    print("Test labels accessed:", False)
    print("Test evaluated:", False)
    print("Medical images saved:", False)
    print("Predicted masks saved:", False)
    print("Allowed for public upload:", False)
    print("OBJECTIVE 4 PRIVATE QUANTITATIVE-XAI SHARD SUCCESSFUL")


if __name__ == "__main__":
    main()
