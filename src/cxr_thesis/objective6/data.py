"""Private PadChest image/report dataset for Objective 6."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cxr_thesis.objective1.features import encode_clinical_features
from cxr_thesis.objective1.preprocessing import load_image

from .text import ReportVocabulary


REQUIRED_COLUMNS = {
    "image_path", "patient_id", "study_id", "report", "age", "sex", "view"
}


def _normalise_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels")
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    return ((np.clip(array, low, high) - low) / (high - low)).astype(np.float32)


class ReportGenerationDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        vocabulary: ReportVocabulary,
        *,
        image_size: int = 320,
        maximum_length: int = 160,
    ) -> None:
        missing = sorted(REQUIRED_COLUMNS - set(manifest.columns))
        if missing:
            raise ValueError(f"Report manifest columns are missing: {missing}")
        if image_size <= 0 or maximum_length < 2:
            raise ValueError("Invalid image or report length")
        self.records = manifest.reset_index(drop=True).to_dict(orient="records")
        self.vocabulary = vocabulary
        self.image_size = int(image_size)
        self.maximum_length = int(maximum_length)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        path = Path(str(record["image_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        image = load_image(path)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = _normalise_image(image)
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        channels = np.repeat(image[None], 3, axis=0)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        channels = ((channels - mean) / std).astype(np.float32)
        clinical = encode_clinical_features(record)
        identifiers = self.vocabulary.encode(
            record["report"], maximum_length=self.maximum_length
        )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(channels)).float(),
            "clinical": torch.tensor(list(clinical.values()), dtype=torch.float32),
            "report_ids": torch.tensor(identifiers, dtype=torch.long),
        }


def collate_reports(
    samples: Sequence[dict[str, torch.Tensor]], *, pad_id: int = 0
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty report batch")
    maximum = max(int(sample["report_ids"].numel()) for sample in samples)
    reports = torch.full((len(samples), maximum), pad_id, dtype=torch.long)
    for row, sample in enumerate(samples):
        values = sample["report_ids"]
        reports[row, : values.numel()] = values
    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "clinical": torch.stack([sample["clinical"] for sample in samples]),
        "report_ids": reports,
    }
