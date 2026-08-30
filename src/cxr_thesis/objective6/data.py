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

from .evaluation import parse_padchest6_labels
from .text import ReportVocabulary

REQUIRED_COLUMNS = {
    "image_path", "patient_id", "study_id", "report", "age", "sex", "view"
}


def select_label_complete_subset(
    manifest: pd.DataFrame,
    size: int,
    *,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic smoke subset containing both classes per target.

    This helper is only for non-research smoke runs. It prevents an arbitrary
    manifest prefix from omitting a rare PadChest target while leaving the full
    locked training and validation cohorts untouched.
    """

    if "labels" not in manifest.columns:
        raise ValueError("A label-complete smoke subset requires labels")
    if size <= 0 or size > len(manifest):
        raise ValueError("Smoke subset size must be in [1, len(manifest)]")
    targets = np.stack(
        manifest["labels"].map(parse_padchest6_labels).to_numpy()
    ).astype(np.int8)
    permutation = np.random.default_rng(seed).permutation(len(manifest))
    selected: list[int] = []
    selected_set: set[int] = set()
    for target in range(targets.shape[1]):
        for value in (1, 0):
            candidates = permutation[targets[permutation, target] == value]
            if not len(candidates):
                raise ValueError(
                    f"The complete manifest is degenerate for target {target}"
                )
            index = int(candidates[0])
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
    if len(selected) > size:
        raise ValueError("Smoke subset is too small for label-complete sampling")
    for index_value in permutation:
        index = int(index_value)
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
        if len(selected) == size:
            break
    subset = manifest.iloc[selected].copy().reset_index(drop=True)
    subset_targets = np.stack(
        subset["labels"].map(parse_padchest6_labels).to_numpy()
    ).astype(np.int8)
    if (subset_targets.sum(axis=0) <= 0).any() or (
        subset_targets.sum(axis=0) >= len(subset_targets)
    ).any():
        raise RuntimeError("Label-complete smoke subset construction failed")
    return subset


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
        include_clinical_labels: bool = False,
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
        self.include_clinical_labels = bool(include_clinical_labels)
        if self.include_clinical_labels and "labels" not in manifest.columns:
            raise ValueError("Clinical-guided report training requires labels")

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
        sample = {
            "image": torch.from_numpy(np.ascontiguousarray(channels)).float(),
            "clinical": torch.tensor(list(clinical.values()), dtype=torch.float32),
            "report_ids": torch.tensor(identifiers, dtype=torch.long),
        }
        if self.include_clinical_labels:
            sample["clinical_labels"] = torch.from_numpy(
                parse_padchest6_labels(record["labels"]).astype(np.float32)
            )
        return sample


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
    batch = {
        "image": torch.stack([sample["image"] for sample in samples]),
        "clinical": torch.stack([sample["clinical"] for sample in samples]),
        "report_ids": reports,
    }
    if all("clinical_labels" in sample for sample in samples):
        batch["clinical_labels"] = torch.stack(
            [sample["clinical_labels"] for sample in samples]
        )
    elif any("clinical_labels" in sample for sample in samples):
        raise ValueError("A report batch cannot mix labelled and unlabelled samples")
    return batch
