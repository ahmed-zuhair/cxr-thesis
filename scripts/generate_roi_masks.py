"""Generate original-resolution ROI masks from a frozen U-Net checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.config import load_config
from cxr_thesis.objective1.manifest import validate_manifest, write_manifest
from cxr_thesis.objective1.preprocessing import load_image, preprocess_cxr, restore_mask
from cxr_thesis.objective1.segmentation import (
    UNet2D,
    probability_uncertainty_metrics,
    remove_small_components,
    validate_roi_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ROI masks with batched, resumable frozen-model inference"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument(
        "--config",
        default=str(REPOSITORY_ROOT / "configs" / "objective1" / "default.yaml"),
    )
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--audit-csv")
    parser.add_argument("--summary-json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--min-component-fraction",
        type=float,
        default=0.001,
        help="Remove components smaller than this fraction of predicted foreground",
    )
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=0,
        help="Absolute model-space component floor, combined with the relative floor",
    )
    parser.add_argument(
        "--uncertainty-margin",
        type=float,
        default=0.10,
        help="Probability distance around the frozen threshold counted as uncertain",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help="Abort unless the checkpoint has this SHA-256 digest",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse masks recorded by a compatible existing output manifest",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace existing mask files; cannot be combined with --resume",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record unreadable cases and continue instead of failing immediately",
    )
    return parser.parse_args()


def resolve(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def safe_name(value: object) -> str:
    text = str(value).strip().replace("/", "_").replace("\\", "_")
    if text in {"", ".", ".."}:
        raise ValueError(f"Unsafe image_id: {value!r}")
    return text


def atomic_write_manifest(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    write_manifest(frame, temporary)
    temporary.replace(path)


def atomic_write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    manifest = Path(args.output_manifest).resolve()
    audit = (
        Path(args.audit_csv).resolve()
        if args.audit_csv
        else manifest.with_name(f"{manifest.stem}_audit.csv")
    )
    summary = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else manifest.with_name(f"{manifest.stem}_summary.json")
    )
    return manifest, audit, summary


def initialise_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    defaults: dict[str, object] = {
        "mask_path": "",
        "mask_model_id": "",
        "mask_checkpoint_sha256": "",
        "mask_threshold": np.nan,
        "mask_postprocessing": "",
        "mask_generation_status": "pending",
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
    string_columns = [
        "mask_path",
        "mask_model_id",
        "mask_checkpoint_sha256",
        "mask_postprocessing",
        "mask_generation_status",
    ]
    for column in string_columns:
        result[column] = result[column].fillna("").astype(str)
    result.loc[result["mask_generation_status"] == "", "mask_generation_status"] = "pending"
    result["mask_threshold"] = pd.to_numeric(result["mask_threshold"], errors="coerce")
    return result


def merge_resume_manifest(frame: pd.DataFrame, previous_path: Path) -> pd.DataFrame:
    previous = pd.read_csv(
        previous_path,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    if previous["image_id"].duplicated().any():
        raise ValueError("The resume manifest contains duplicate image_id values")
    if set(previous["image_id"]) != set(frame["image_id"]):
        raise ValueError("The resume manifest does not describe the same image set")
    previous = previous.set_index("image_id")
    result = frame.set_index("image_id")
    resume_columns = [column for column in previous.columns if column.startswith("mask_")]
    for column in resume_columns:
        result[column] = previous[column].reindex(result.index)
    return result.reset_index()


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(
        temporary,
        format="PNG",
    )
    temporary.replace(path)


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.resume and args.overwrite_existing:
        raise ValueError("--resume and --overwrite-existing cannot be combined")
    if not 0.0 < args.uncertainty_margin < 0.5:
        raise ValueError("--uncertainty-margin must be between 0 and 0.5")

    config = load_config(args.config)
    root = Path(args.data_root)
    frame = pd.read_csv(
        args.manifest,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    validate_manifest(frame, require_files=True, root=root)
    frame = initialise_output_columns(frame)

    output_manifest, audit_path, summary_path = output_paths(args)
    if args.resume and output_manifest.is_file():
        frame = merge_resume_manifest(frame, output_manifest)
        frame = initialise_output_columns(frame)

    checkpoint_path = Path(args.checkpoint).resolve()
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if args.expected_checkpoint_sha256 and digest.lower() != args.expected_checkpoint_sha256.lower():
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {args.expected_checkpoint_sha256}, received {digest}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "UNet2D":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint.get('architecture')!r}")
    channels = tuple(int(value) for value in checkpoint.get("channels", (32, 64, 128, 256)))
    model = UNet2D(channels=channels)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(
        checkpoint.get("validation_metrics", {}).get(
            "threshold", config.segmentation.threshold
        )
    )
    postprocessing_signature = (
        f"relative-area>={args.min_component_fraction:g};"
        f"min-pixels={args.min_component_pixels};"
        "fixed-component-count=false;"
        f"uncertainty-margin={args.uncertainty_margin:g}"
    )
    device = select_device(args.device)
    model = model.to(device).eval()
    mask_directory = Path(args.mask_dir).resolve()
    mask_directory.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    selected_indices = list(frame.index)
    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]

    audit_by_id: dict[str, dict[str, object]] = {}
    if args.resume and audit_path.is_file():
        existing_audit = pd.read_csv(audit_path)
        if "image_id" in existing_audit.columns:
            audit_by_id = {
                str(row["image_id"]): row.to_dict()
                for _, row in existing_audit.iterrows()
            }

    pending: list[int] = []
    skipped = 0
    for index in selected_indices:
        target = mask_directory / f"{safe_name(frame.at[index, 'image_id'])}.png"
        compatible = (
            args.resume
            and str(frame.at[index, "mask_checkpoint_sha256"]) == digest
            and str(frame.at[index, "mask_postprocessing"]) == postprocessing_signature
            and Path(str(frame.at[index, "mask_path"])).is_file()
        )
        if compatible:
            frame.at[index, "mask_generation_status"] = "complete"
            skipped += 1
            continue
        if target.exists() and not args.overwrite_existing:
            raise FileExistsError(
                f"Mask already exists but is not resume-compatible: {target}. "
                "Use a new output directory, --resume with the matching manifest, "
                "or --overwrite-existing."
            )
        pending.append(index)

    print(
        json.dumps(
            {
                "event": "start",
                "manifest_rows": int(len(frame)),
                "selected": int(len(selected_indices)),
                "pending": int(len(pending)),
                "resumed": int(skipped),
                "batch_size": int(args.batch_size),
                "device": str(device),
                "checkpoint_sha256": digest,
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "threshold": threshold,
                "postprocessing": postprocessing_signature,
            }
        )
    )

    generated = 0
    failures = 0
    start_time = time.perf_counter()
    last_saved = 0
    for batch_start in range(0, len(pending), args.batch_size):
        batch_indices = pending[batch_start : batch_start + args.batch_size]
        loaded: list[tuple[int, np.ndarray, object, Path]] = []
        for index in batch_indices:
            image_id = str(frame.at[index, "image_id"])
            target = mask_directory / f"{safe_name(image_id)}.png"
            try:
                image = load_image(resolve(frame.at[index, "image_path"], root))
                processed, geometry = preprocess_cxr(image, config.preprocessing)
                loaded.append((index, processed, geometry, target))
            except Exception as error:
                failures += 1
                frame.at[index, "mask_generation_status"] = "failed"
                audit_by_id[image_id] = {
                    "image_id": image_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                if not args.continue_on_error:
                    raise

        if loaded:
            tensor = torch.from_numpy(
                np.stack([item[1] for item in loaded]).astype(np.float32) / 255.0
            )[:, None].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(tensor)
                probabilities = torch.sigmoid(logits)[:, 0].float().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_inference_seconds = time.perf_counter() - inference_start

            for position, (index, _, geometry, target) in enumerate(loaded):
                image_id = str(frame.at[index, "image_id"])
                raw_mask = probabilities[position] >= threshold
                uncertainty = probability_uncertainty_metrics(
                    probabilities[position],
                    threshold=threshold,
                    margin=args.uncertainty_margin,
                )
                cleaned_mask, cleanup = remove_small_components(
                    raw_mask,
                    min_component_fraction=args.min_component_fraction,
                    min_component_pixels=args.min_component_pixels,
                )
                original_mask = restore_mask(cleaned_mask, geometry)
                save_mask(original_mask, target)
                quality = validate_roi_mask(cleaned_mask, config.segmentation)
                output_components = max(
                    0,
                    int(
                        cv2.connectedComponents(
                            original_mask.astype(np.uint8), connectivity=8
                        )[0]
                        - 1
                    ),
                )

                frame.at[index, "mask_path"] = str(target)
                frame.at[index, "mask_model_id"] = "UNet2D-union-lung-roi"
                frame.at[index, "mask_checkpoint_sha256"] = digest
                frame.at[index, "mask_threshold"] = threshold
                frame.at[index, "mask_postprocessing"] = postprocessing_signature
                frame.at[index, "mask_generation_status"] = "complete"
                audit_by_id[image_id] = {
                    "image_id": image_id,
                    "status": "complete",
                    "split": str(frame.at[index, "split"]),
                    "original_height": int(geometry.original_height),
                    "original_width": int(geometry.original_width),
                    "model_size": int(config.preprocessing.image_size),
                    "threshold": threshold,
                    "roi_fraction_model_space": float(quality["roi_fraction"]),
                    "roi_fraction_original_space": float(original_mask.mean()),
                    "touches_border_model_space": bool(quality["touches_border"]),
                    "is_nonempty": bool(quality["is_nonempty"]),
                    "is_plausible": bool(quality["is_plausible"]),
                    "components_before_cleanup": int(cleanup["components_before"]),
                    "components_after_cleanup": int(cleanup["components_after"]),
                    "components_original_space": output_components,
                    "removed_pixels_model_space": int(cleanup["removed_pixels"]),
                    "removed_fraction_of_prediction": float(cleanup["removed_fraction"]),
                    **uncertainty,
                    "inference_seconds_share": float(batch_inference_seconds / len(loaded)),
                    "mask_path": str(target),
                    "checkpoint_sha256": digest,
                    "postprocessing": postprocessing_signature,
                    "error": "",
                }
                generated += 1

        completed_now = generated + failures
        elapsed = time.perf_counter() - start_time
        print(
            json.dumps(
                {
                    "event": "progress",
                    "selected_completed": int(skipped + completed_now),
                    "selected_total": int(len(selected_indices)),
                    "generated": int(generated),
                    "resumed": int(skipped),
                    "failed": int(failures),
                    "elapsed_seconds": float(elapsed),
                    "generated_per_second": float(generated / elapsed) if elapsed else 0.0,
                }
            )
        )
        if completed_now - last_saved >= args.save_every:
            atomic_write_manifest(frame, output_manifest)
            atomic_write_table(pd.DataFrame(audit_by_id.values()), audit_path)
            last_saved = completed_now

    atomic_write_manifest(frame, output_manifest)
    audit_frame = pd.DataFrame(audit_by_id.values())
    if not audit_frame.empty:
        order = {str(image_id): position for position, image_id in enumerate(frame["image_id"])}
        audit_frame["_order"] = audit_frame["image_id"].astype(str).map(order)
        audit_frame = audit_frame.sort_values("_order").drop(columns="_order")
    atomic_write_table(audit_frame, audit_path)

    elapsed = time.perf_counter() - start_time
    completed_mask_count = int(
        (frame["mask_generation_status"].astype(str) == "complete").sum()
    )
    if not audit_frame.empty and "status" in audit_frame.columns:
        complete_audit = audit_frame[audit_frame["status"] == "complete"]
    else:
        complete_audit = pd.DataFrame()
    summary: dict[str, object] = {
        "manifest": str(Path(args.manifest).resolve()),
        "output_manifest": str(output_manifest),
        "audit_csv": str(audit_path),
        "mask_directory": str(mask_directory),
        "manifest_rows": int(len(frame)),
        "selected_cases": int(len(selected_indices)),
        "generated_this_run": int(generated),
        "resumed_this_run": int(skipped),
        "failed_this_run": int(failures),
        "completed_masks_in_output_manifest": completed_mask_count,
        "elapsed_seconds": float(elapsed),
        "generated_per_second": float(generated / elapsed) if elapsed else 0.0,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "batch_size": int(args.batch_size),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "threshold": threshold,
        "postprocessing": postprocessing_signature,
        "uncertainty_margin": float(args.uncertainty_margin),
        "mean_roi_fraction_model_space": (
            float(complete_audit["roi_fraction_model_space"].mean())
            if not complete_audit.empty
            else None
        ),
        "mean_removed_fraction_of_prediction": (
            float(complete_audit["removed_fraction_of_prediction"].mean())
            if not complete_audit.empty
            else None
        ),
        "mean_uncertain_fraction": (
            float(complete_audit["uncertain_fraction"].mean())
            if not complete_audit.empty
            else None
        ),
        "mean_binary_entropy": (
            float(complete_audit["mean_binary_entropy"].mean())
            if not complete_audit.empty
            else None
        ),
        "mean_boundary_entropy": (
            float(complete_audit["boundary_entropy_mean"].mean())
            if not complete_audit.empty
            else None
        ),
        "nonempty_rate": (
            float(complete_audit["is_nonempty"].astype(bool).mean())
            if not complete_audit.empty
            else None
        ),
        "plausible_rate": (
            float(complete_audit["is_plausible"].astype(bool).mean())
            if not complete_audit.empty
            else None
        ),
    }
    atomic_write_json(summary, summary_path)
    print(json.dumps({"event": "complete", **summary}, indent=2))


if __name__ == "__main__":
    main()
