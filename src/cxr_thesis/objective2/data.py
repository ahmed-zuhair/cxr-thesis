"""Patient-level manifest datasets for Objective 2 classification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cxr_thesis.objective1.features import encode_clinical_features
from cxr_thesis.objective1.graphs import GraphSample
from cxr_thesis.objective1.preprocessing import load_image


def _safe_name(value: object) -> str:
    text = str(value).strip().replace("/", "_").replace("\\", "_")
    if text in {"", ".", ".."}:
        raise ValueError(f"Unsafe image identifier: {value!r}")
    return text


def _resolve_image(path_value: object, data_root: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else data_root / path


def _labels(record: dict[str, object], columns: Sequence[str]) -> torch.Tensor:
    values = np.asarray([record[column] for column in columns], dtype=np.float32)
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Classification labels must be binary 0/1")
    return torch.from_numpy(values)


def _clinical(record: dict[str, object]) -> torch.Tensor:
    encoded = encode_clinical_features(record)
    return torch.tensor(list(encoded.values()), dtype=torch.float32)


def _normalise_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels")
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    return ((np.clip(array, low, high) - low) / (high - low)).astype(np.float32)


class ImageClassificationDataset(Dataset):
    """Load one CXR, clinical vector, and multi-hot label vector per row."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        label_columns: Sequence[str],
        *,
        data_root: str | Path = ".",
        image_size: int = 224,
        augment: bool = False,
        seed: int = 42,
        augmentation_profile: str = "baseline",
        epoch_varying_augmentation: bool = False,
        output_channels: int = 1,
        normalisation: str = "unit",
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        missing = sorted({"image_path", *label_columns} - set(manifest.columns))
        if missing:
            raise ValueError(f"Manifest columns are missing: {missing}")
        self.records = manifest.reset_index(drop=True).to_dict(orient="records")
        self.label_columns = list(label_columns)
        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        if augmentation_profile not in {
            "baseline",
            "cxr_mild",
            "objective5_locked",
        }:
            raise ValueError(
                "augmentation_profile must be baseline, cxr_mild, or objective5_locked"
            )
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if output_channels not in {1, 3}:
            raise ValueError("output_channels must be one or three")
        if normalisation not in {"unit", "imagenet"}:
            raise ValueError("normalisation must be unit or imagenet")
        if normalisation == "imagenet" and output_channels != 3:
            raise ValueError("ImageNet normalisation requires three output channels")
        self.augmentation_profile = augmentation_profile
        self.epoch_varying_augmentation = bool(epoch_varying_augmentation)
        self.output_channels = int(output_channels)
        self.normalisation = normalisation
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic but epoch-varying augmentation stream."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        if not self.epoch_varying_augmentation:
            # Preserve the exact frozen Objective 2 baseline augmentation stream.
            return np.random.default_rng(self.seed + index)
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, index])
        )

    def _augment(self, image: np.ndarray, index: int) -> np.ndarray:
        rng = self._rng(index)
        result = np.asarray(image, dtype=np.float32)
        if rng.random() < self.horizontal_flip_probability:
            result = np.fliplr(result).copy()
        if self.augmentation_profile in {"cxr_mild", "objective5_locked"}:
            height, width = result.shape
            locked = self.augmentation_profile == "objective5_locked"
            angle = float(rng.uniform(-5.0, 5.0) if locked else rng.uniform(-7.0, 7.0))
            scale = 1.0 if locked else float(rng.uniform(0.95, 1.05))
            matrix = cv2.getRotationMatrix2D(
                ((width - 1) / 2.0, (height - 1) / 2.0), angle, scale
            )
            matrix[0, 2] += float(rng.uniform(-0.03, 0.03) * width)
            matrix[1, 2] += float(rng.uniform(-0.03, 0.03) * height)
            result = cv2.warpAffine(
                result,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            if locked:
                brightness_factor = float(rng.uniform(0.90, 1.10))
                result = np.clip(result * brightness_factor, 0.0, 1.0)
                contrast = float(rng.uniform(0.90, 1.10))
                brightness = 0.0
            else:
                gamma = float(rng.uniform(0.90, 1.10))
                result = np.power(np.clip(result, 0.0, 1.0), gamma)
                noise_sigma = float(rng.uniform(0.0, 0.015))
                if noise_sigma > 0.0:
                    result = result + rng.normal(0.0, noise_sigma, result.shape)
                contrast = float(rng.uniform(0.85, 1.15))
                brightness = float(rng.uniform(-0.05, 0.05))
        else:
            contrast = float(rng.uniform(0.90, 1.10))
            brightness = float(rng.uniform(-0.05, 0.05))
        return np.clip((result - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0).astype(
            np.float32
        )

    def _format_channels(self, image: np.ndarray) -> np.ndarray:
        if self.output_channels == 1:
            return image[None]
        channels = np.repeat(image[None], 3, axis=0)
        if self.normalisation == "imagenet":
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            channels = (channels - mean) / std
        return channels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        image = load_image(_resolve_image(record["image_path"], self.data_root))
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = _normalise_image(image)
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        if self.augment:
            image = self._augment(image, index)
        image = self._format_channels(image)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "clinical": _clinical(record),
            "labels": _labels(record, self.label_columns),
        }


@dataclass
class GraphBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch_index: torch.Tensor
    clinical: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            batch_index=self.batch_index.to(device),
            clinical=self.clinical.to(device),
            labels=self.labels.to(device),
        )


class GraphClassificationDataset(Dataset):
    """Load precomputed ROI patch graphs without requiring PyG."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        label_columns: Sequence[str],
        graph_root: str | Path,
    ) -> None:
        missing = sorted({"image_id", *label_columns} - set(manifest.columns))
        if missing:
            raise ValueError(f"Manifest columns are missing: {missing}")
        self.records = manifest.reset_index(drop=True).to_dict(orient="records")
        self.label_columns = list(label_columns)
        self.graph_root = Path(graph_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        graph = GraphSample.load(
            self.graph_root / f"{_safe_name(record['image_id'])}.npz"
        )
        return {
            "x": torch.from_numpy(graph.x).float(),
            "edge_index": torch.from_numpy(graph.edge_index).long(),
            "clinical": _clinical(record),
            "labels": _labels(record, self.label_columns),
        }


def collate_graph_samples(samples: list[dict[str, torch.Tensor]]) -> GraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty graph batch")
    node_features: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []
    clinical: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    offset = 0
    feature_dimension = int(samples[0]["x"].shape[1])
    for graph_index, sample in enumerate(samples):
        x = sample["x"]
        if x.ndim != 2 or x.shape[1] != feature_dimension:
            raise ValueError("Graph node-feature dimensions must match")
        node_features.append(x)
        edges.append(sample["edge_index"] + offset)
        batch_indices.append(torch.full((x.shape[0],), graph_index, dtype=torch.long))
        clinical.append(sample["clinical"])
        labels.append(sample["labels"])
        offset += int(x.shape[0])
    return GraphBatch(
        x=torch.cat(node_features, dim=0),
        edge_index=torch.cat(edges, dim=1),
        batch_index=torch.cat(batch_indices, dim=0),
        clinical=torch.stack(clinical),
        labels=torch.stack(labels),
    )
