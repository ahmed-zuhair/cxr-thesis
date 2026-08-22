"""Generate original-resolution ROI masks and update a manifest copy."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

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
from cxr_thesis.objective1.segmentation import UNet2D, predict_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ROI masks from a frozen U-Net")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--config", default=str(REPOSITORY_ROOT / "configs" / "objective1" / "default.yaml"))
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def resolve(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def safe_name(value: object) -> str:
    text = str(value).strip().replace("/", "_").replace("\\", "_")
    if text in {"", ".", ".."}:
        raise ValueError(f"Unsafe image_id: {value!r}")
    return text


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(args.data_root)
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str, "study_id": str, "image_id": str})
    validate_manifest(frame, require_files=True, root=root)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = UNet2D()
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint.get("validation_metrics", {}).get("threshold", config.segmentation.threshold))
    segmentation_config = type(config.segmentation)(
        threshold=threshold,
        min_roi_fraction=config.segmentation.min_roi_fraction,
        max_roi_fraction=config.segmentation.max_roi_fraction,
        keep_largest_components=config.segmentation.keep_largest_components,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.mask_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    selected_indices = frame.index if args.limit is None else frame.index[: args.limit]
    for index in selected_indices:
        image = load_image(resolve(frame.at[index, "image_path"], root))
        processed, geometry = preprocess_cxr(image, config.preprocessing)
        model_mask = predict_mask(model, processed, segmentation_config, device=device)
        original_mask = restore_mask(model_mask, geometry)
        target = output_dir / f"{safe_name(frame.at[index, 'image_id'])}.png"
        Image.fromarray(original_mask.astype(np.uint8) * 255).save(target)
        frame.at[index, "mask_path"] = str(target)
        frame.at[index, "mask_model_id"] = "UNet2D-union-roi"
        frame.at[index, "mask_checkpoint_sha256"] = digest
    write_manifest(frame, args.output_manifest)
    print(f"Generated {len(selected_indices)} masks; wrote {args.output_manifest}")


if __name__ == "__main__":
    main()

