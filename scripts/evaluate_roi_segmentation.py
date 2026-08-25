"""Prediction-blind evaluation of a frozen ROI checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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

from cxr_thesis.objective1.config import load_config
from cxr_thesis.objective1.manifest import validate_manifest
from cxr_thesis.objective1.segmentation import (
    UNet2D,
    dice_score,
    hausdorff95,
    iou_score,
)
from cxr_thesis.objective1.segmentation_data import ROISegmentationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen ROI checkpoint without test-set tuning"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs" / "objective1" / "default.yaml"),
    )
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def bootstrap_mean_ci(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(samples, finite.size))
    means = finite[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def aggregate(frame: pd.DataFrame, *, samples: int, seed: int) -> dict[str, object]:
    if frame.empty:
        return {"cases": 0}
    result: dict[str, object] = {"cases": int(len(frame))}
    for offset, column in enumerate(("dice", "iou", "hd95_pixels_224")):
        values = frame[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        mean = float(finite.mean()) if finite.size else float("nan")
        lower, upper = bootstrap_mean_ci(
            values, samples=samples, seed=seed + offset
        )
        result[f"{column}_mean"] = mean
        result[f"{column}_bootstrap_95ci"] = [lower, upper]
    result["precision_mean"] = float(frame["precision"].mean())
    result["recall_mean"] = float(frame["recall"].mean())
    result["empty_failure_rate"] = float((~np.isfinite(frame["hd95_pixels_224"])).mean())
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = pd.read_csv(
        args.manifest,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
        keep_default_na=False,
    )
    validate_manifest(manifest, require_files=True, root=args.data_root)
    split_frame = manifest[manifest["split"].astype(str) == args.split].copy()
    if split_frame.empty:
        raise ValueError(f"No cases found for split {args.split!r}")

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_hash = sha256(checkpoint_path)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_hash.lower() != args.expected_checkpoint_sha256.lower()
    ):
        raise ValueError(
            "Checkpoint SHA-256 mismatch: expected "
            f"{args.expected_checkpoint_sha256}, received {checkpoint_hash}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "UNet2D":
        raise ValueError("The checkpoint is not an Objective 1 UNet2D")
    validation_metrics = checkpoint.get("validation_metrics", {})
    if "threshold" not in validation_metrics:
        raise ValueError("Checkpoint lacks a validation-selected threshold")
    threshold = float(validation_metrics["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("The frozen validation threshold is invalid")

    channels = tuple(int(value) for value in checkpoint.get("channels", (32, 64, 128, 256)))
    model = UNet2D(channels=channels)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    config = load_config(args.config)
    dataset = ROISegmentationDataset(
        manifest,
        args.data_root,
        config.preprocessing,
        split=args.split,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    metadata = split_frame.set_index("image_id")
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for images, masks, image_ids in loader:
            probabilities = torch.sigmoid(model(images.to(device)))[:, 0].cpu().numpy()
            targets = masks[:, 0].numpy() > 0
            for probability, target, image_id_value in zip(
                probabilities, targets, image_ids
            ):
                image_id = str(image_id_value)
                prediction = probability >= threshold
                true_positive = int(np.logical_and(prediction, target).sum())
                false_positive = int(np.logical_and(prediction, ~target).sum())
                false_negative = int(np.logical_and(~prediction, target).sum())
                precision = true_positive / max(true_positive + false_positive, 1)
                recall = true_positive / max(true_positive + false_negative, 1)
                meta = metadata.loc[image_id]
                rows.append(
                    {
                        "image_id": image_id,
                        "dice": dice_score(prediction, target),
                        "iou": iou_score(prediction, target),
                        "hd95_pixels_224": hausdorff95(prediction, target),
                        "precision": float(precision),
                        "recall": float(recall),
                        "annotation_progress_status": str(
                            meta.get("annotation_progress_status", "")
                        ),
                        "annotation_qc_clean": str(
                            meta.get("annotation_qc_clean", "")
                        ).lower()
                        in {"true", "1", "yes"},
                    }
                )

    per_case = pd.DataFrame(rows)
    all_metrics = aggregate(
        per_case, samples=args.bootstrap_samples, seed=args.seed
    )
    clean = per_case[per_case["annotation_qc_clean"]].copy()
    complete = per_case[
        per_case["annotation_progress_status"] == "complete"
    ].copy()
    summary = {
        "artifact": "Frozen prediction-blind ROI segmentation evaluation",
        "split": args.split,
        "checkpoint_sha256": checkpoint_hash,
        "parent_checkpoint_sha256": checkpoint.get("parent_checkpoint_sha256"),
        "training_role": checkpoint.get("training_role", "unknown"),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "frozen_validation_threshold": threshold,
        "test_threshold_tuning": False,
        "test_used_for_model_selection": False,
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "all_locked_cases_primary": all_metrics,
        "progress_complete_sensitivity": aggregate(
            complete, samples=args.bootstrap_samples, seed=args.seed + 100
        ),
        "qc_clean_sensitivity": aggregate(
            clean, samples=args.bootstrap_samples, seed=args.seed + 200
        ),
        "annotation_progress_needs_second_review_cases": int(
            (per_case["annotation_progress_status"] == "needs_review").sum()
        ),
        "annotation_qc_flagged_cases": int((~per_case["annotation_qc_clean"]).sum()),
        "annotation_limitation_declared": bool((~per_case["annotation_qc_clean"]).any()),
        "patient_or_image_identifiers_included": False,
        "medical_images_included": False,
        "annotation_masks_included": False,
        "private_per_case_metrics_published": False,
    }
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    private_dir = output / "private"
    public_dir = output / "public"
    private_dir.mkdir(parents=True)
    public_dir.mkdir(parents=True)
    private_path = private_dir / "locked_test_per_case_private.csv"
    per_case.to_csv(private_path, index=False, lineterminator="\n")
    summary_path = public_dir / "locked_test_summary_public.json"
    write_lf(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_lf(
        summary_path.with_suffix(summary_path.suffix + ".sha256"),
        f"{sha256(summary_path)}  {summary_path.name}\n",
    )
    print("--- FROZEN LOCKED-TEST EVALUATION ---")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Private per-case metrics: {private_path}")
    print(f"Public summary: {summary_path}")
    print("Identifiers displayed: False")
    print("Test-set tuning performed: False")


if __name__ == "__main__":
    main()
