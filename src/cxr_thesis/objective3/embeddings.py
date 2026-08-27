"""Private, order-preserving embedding shards for Objective 3."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def save_embedding_shard(
    path: str | Path,
    embeddings: np.ndarray,
    image_ids: Sequence[str],
) -> Path:
    """Atomically save finite 160-dimensional embeddings and private IDs."""

    target = Path(path)
    values = np.asarray(embeddings, dtype=np.float32)
    identifiers = np.asarray([str(value) for value in image_ids], dtype=str)
    if values.ndim != 2 or values.shape[1] != 160:
        raise ValueError("Embeddings must have shape [cases, 160]")
    if identifiers.shape != (values.shape[0],):
        raise ValueError("One image ID is required for each embedding")
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("Embedding image IDs must be unique")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings must be finite")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, embeddings=values, image_ids=identifiers)
    os.replace(temporary, target)
    return target


def load_embedding_shard(
    path: str | Path,
    *,
    expected_image_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a private embedding shard."""

    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != {"embeddings", "image_ids"}:
            raise ValueError("Embedding shard contains unexpected arrays")
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        image_ids = np.asarray(archive["image_ids"], dtype=str)
    if embeddings.ndim != 2 or embeddings.shape[1] != 160:
        raise ValueError("Embeddings must have shape [cases, 160]")
    if image_ids.shape != (embeddings.shape[0],):
        raise ValueError("Embedding identifiers are misaligned")
    if len(set(image_ids.tolist())) != len(image_ids):
        raise ValueError("Embedding image IDs must be unique")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite")
    if expected_image_ids is not None:
        expected = np.asarray([str(value) for value in expected_image_ids], dtype=str)
        if not np.array_equal(image_ids, expected):
            raise ValueError("Embedding IDs do not match the frozen manifest order")
    return embeddings, image_ids
