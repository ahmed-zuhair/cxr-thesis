"""Patient-level manifest datasets for Objective 2 classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
            rng = np.random.default_rng(self.seed + index)
            if rng.random() < 0.5:
                image = np.fliplr(image).copy()
            contrast = float(rng.uniform(0.90, 1.10))
            brightness = float(rng.uniform(-0.05, 0.05))
            image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image[None])).float(),
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

    def to(self, device: torch.device | str) -> "GraphBatch":
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
