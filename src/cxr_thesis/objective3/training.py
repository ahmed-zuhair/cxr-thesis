"""Shared, paired training utilities for Objective 3 bottleneck heads."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .embeddings import load_embedding_shard


def load_embedding_corpus(
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    recovery_index: dict[str, object],
    shard_root: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all shards in frozen manifest order and return train/validation arrays."""

    combined = pd.concat([train_manifest, validation_manifest], ignore_index=True)
    records = sorted(
        recovery_index.get("shards", []), key=lambda record: int(record["start"])
    )
    if not records:
        raise ValueError("Embedding recovery index has no shards")
    expected_start = 0
    arrays: list[np.ndarray] = []
    root = Path(shard_root)
    for record in records:
        start = int(record["start"])
        stop = int(record["stop"])
        if start != expected_start or stop <= start or stop > len(combined):
            raise ValueError("Embedding shards are not contiguous")
        expected_ids = combined.iloc[start:stop]["image_id"].astype(str).tolist()
        shard_path = root / f"{record['shard']}.npz"
        embeddings, _ = load_embedding_shard(
            shard_path, expected_image_ids=expected_ids
        )
        if len(embeddings) != stop - start:
            raise ValueError("Embedding shard case count does not match its index")
        arrays.append(embeddings)
        expected_start = stop
    if expected_start != len(combined):
        raise ValueError("Embedding index does not cover every manifest row")
    all_embeddings = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    train_cases = len(train_manifest)
    return all_embeddings[:train_cases], all_embeddings[train_cases:]


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise scaling on training embeddings only."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 160 or not np.isfinite(array).all():
        raise ValueError("Training embeddings must be finite [cases, 160]")
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = array.std(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviation = np.maximum(standard_deviation, np.float32(1e-6))
    return mean, standard_deviation


def apply_standardizer(
    values: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> np.ndarray:
    """Apply the frozen training-only scaling transformation."""

    array = np.asarray(values, dtype=np.float32)
    result = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(
        standard_deviation, dtype=np.float32
    )
    if result.shape != array.shape or not np.isfinite(result).all():
        raise ValueError("Standardized embeddings are invalid")
    return result.astype(np.float32, copy=False)


def make_loader(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[DataLoader, torch.Generator]:
    """Build a deterministic in-memory embedding loader."""

    features = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    targets = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    if features.ndim != 2 or features.shape[1] != 160:
        raise ValueError("Embedding tensor must have shape [cases, 160]")
    if targets.ndim != 2 or targets.shape[0] != features.shape[0]:
        raise ValueError("Label tensor must align with embeddings")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )
    return loader, generator


def initialize_shared_layers(model: nn.Module, seed: int) -> None:
    """Give both heads identical shared projection/classifier initialization."""

    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    layers = [model.input_projection, model.classifier[-1]]
    with torch.no_grad():
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight, generator=generator)
            nn.init.zeros_(layer.bias)


def shared_layer_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached copy of parameters shared by both paired heads."""

    return {
        "input_projection.weight": model.input_projection.weight.detach().cpu().clone(),
        "input_projection.bias": model.input_projection.bias.detach().cpu().clone(),
        "classifier.weight": model.classifier[-1].weight.detach().cpu().clone(),
        "classifier.bias": model.classifier[-1].bias.detach().cpu().clone(),
    }


def positive_weights(labels: np.ndarray) -> np.ndarray:
    """Calculate the same raw training-only BCE weights used by the GAT baseline."""

    targets = np.asarray(labels, dtype=np.float32)
    positives = targets.sum(axis=0)
    if np.any(positives <= 0) or np.any(positives >= len(targets)):
        raise ValueError("Every training label requires positive and negative cases")
    return ((len(targets) - positives) / positives).astype(np.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_cases = 0
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(embeddings)
        total_cases += len(embeddings)
    return total_loss / max(1, total_cases)


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for embeddings, labels in loader:
        logits = model(embeddings.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def labels_from_manifest(
    frame: pd.DataFrame, label_names: Sequence[str]
) -> np.ndarray:
    columns = [f"label_{label}" for label in label_names]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest labels are missing: {missing}")
    labels = frame[columns].to_numpy(dtype=np.float32)
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("Manifest labels must be binary")
    return labels
