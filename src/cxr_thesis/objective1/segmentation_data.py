"""Manifest-backed ROI segmentation dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PreprocessingConfig
from .preprocessing import load_image, preprocess_cxr, transform_mask


def _resolve(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


class ROISegmentationDataset(Dataset):
    """Load CXR/union-mask pairs from the canonical manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        data_root: str | Path,
        preprocessing: PreprocessingConfig,
        *,
        split: str,
        augment: bool = False,
    ) -> None:
        required = {"image_id", "image_path", "mask_path", "split"}
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(f"Segmentation manifest is missing columns: {missing}")
        self.frame = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No segmentation cases found for split '{split}'")
        if self.frame["mask_path"].isna().any() or (self.frame["mask_path"].astype(str).str.strip() == "").any():
            raise ValueError(f"Every {split} segmentation case must have mask_path")
        self.root = Path(data_root)
        self.preprocessing = preprocessing
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image = load_image(_resolve(row["image_path"], self.root))
        mask = load_image(_resolve(row["mask_path"], self.root))
        if mask.ndim == 3:
            mask = mask[..., 0]
        processed, geometry = preprocess_cxr(image, self.preprocessing)
        transformed_mask = transform_mask(mask, geometry)
        if self.augment and np.random.random() < 0.5:
            processed = np.ascontiguousarray(np.fliplr(processed))
            transformed_mask = np.ascontiguousarray(np.fliplr(transformed_mask))
        image_tensor = torch.from_numpy(processed.astype(np.float32) / 255.0)[None]
        mask_tensor = torch.from_numpy((transformed_mask > 0).astype(np.float32))[None]
        return image_tensor, mask_tensor, str(row["image_id"])

