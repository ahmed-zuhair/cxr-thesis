"""Dependency-light 2D, 3D, and multimodal graph construction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass
class GraphSample:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    node_type: np.ndarray
    node_position: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.edge_index = np.asarray(self.edge_index, dtype=np.int64)
        self.edge_attr = np.asarray(self.edge_attr, dtype=np.float32)
        self.node_type = np.asarray(self.node_type, dtype=str)
        self.node_position = np.asarray(self.node_position, dtype=np.float32)
        if self.x.ndim != 2 or self.x.shape[0] == 0:
            raise ValueError("x must be a non-empty [nodes, features] matrix")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if self.edge_attr.ndim != 2 or self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_attr rows must match edge count")
        if self.node_type.shape != (self.x.shape[0],):
            raise ValueError("node_type must contain one value per node")
        if self.node_position.ndim != 2 or self.node_position.shape[0] != self.x.shape[0]:
            raise ValueError("node_position must contain one row per node")
        if self.edge_index.size:
            if self.edge_index.min() < 0 or self.edge_index.max() >= self.x.shape[0]:
                raise ValueError("edge_index references a nonexistent node")
        if not np.isfinite(self.x).all() or not np.isfinite(self.edge_attr).all():
            raise ValueError("Graph features must be finite")

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_type=self.node_type,
            node_position=self.node_position,
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "GraphSample":
        with np.load(path, allow_pickle=False) as archive:
            sample = cls(
                x=archive["x"],
                edge_index=archive["edge_index"],
                edge_attr=archive["edge_attr"],
                node_type=archive["node_type"],
                node_position=archive["node_position"],
                metadata=json.loads(str(archive["metadata_json"])),
            )
        sample.validate()
        return sample

    def to_pyg(self):
        """Convert to PyTorch Geometric when that optional extra is installed."""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError("Install the 'graph' extra for PyTorch Geometric") from exc
        self.validate()
        return Data(
            x=torch.from_numpy(self.x),
            edge_index=torch.from_numpy(self.edge_index),
            edge_attr=torch.from_numpy(self.edge_attr),
            node_type=list(self.node_type),
            pos=torch.from_numpy(self.node_position),
            metadata=self.metadata,
        )


def _axis_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    if parts <= 0 or parts > length:
        raise ValueError(f"Grid parts {parts} must be within [1, {length}]")
    values = np.linspace(0, length, parts + 1).round().astype(int)
    return [(int(values[i]), int(values[i + 1])) for i in range(parts)]


def _patch_features(values: np.ndarray, mask: np.ndarray, position: tuple[float, ...]) -> np.ndarray:
    selected = values[mask]
    if selected.size == 0:
        stats = [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        stats = [
            float(selected.mean()),
            float(selected.std()),
            float(selected.min()),
            float(selected.max()),
            float(mask.mean()),
        ]
    return np.asarray(stats + list(position), dtype=np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)


def _add_knn_edges(features: np.ndarray, existing: set[tuple[int, int]], k: int) -> None:
    if k <= 0 or len(features) <= 1:
        return
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalised = features / np.maximum(norms, 1e-8)
    similarity = normalised @ normalised.T
    np.fill_diagonal(similarity, -np.inf)
    actual_k = min(k, len(features) - 1)
    neighbours = np.argpartition(-similarity, kth=actual_k - 1, axis=1)[:, :actual_k]
    for source, targets in enumerate(neighbours):
        for target in targets:
            existing.add((int(source), int(target)))


def _finalise_graph(
    features: list[np.ndarray],
    positions: list[tuple[float, ...]],
    node_types: list[str],
    edges: set[tuple[int, int]],
    spatial_edges: set[tuple[int, int]],
    metadata: dict[str, object],
) -> GraphSample:
    x = np.stack(features).astype(np.float32)
    pos = np.asarray(positions, dtype=np.float32)
    ordered = sorted(edges)
    edge_index = np.asarray(ordered, dtype=np.int64).T if ordered else np.empty((2, 0), dtype=np.int64)
    edge_features: list[list[float]] = []
    for source, target in ordered:
        delta = pos[target] - pos[source]
        distance = float(np.linalg.norm(delta))
        similarity = _cosine(x[source], x[target])
        edge_features.append(
            [distance, similarity, float((source, target) in spatial_edges)] + delta.tolist()
        )
    edge_attr = np.asarray(edge_features, dtype=np.float32)
    if not ordered:
        edge_attr = np.empty((0, 3 + pos.shape[1]), dtype=np.float32)
    graph = GraphSample(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=np.asarray(node_types),
        node_position=pos,
        metadata=metadata,
    )
    graph.validate()
    return graph


def build_patch_graph_2d(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    grid: tuple[int, int] = (7, 7),
    connectivity: int = 8,
    knn_k: int = 3,
    include_empty_nodes: bool = False,
) -> GraphSample:
    """Build an ROI-aware image patch graph with spatial and kNN edges."""
    values = np.asarray(image, dtype=np.float32)
    binary = np.asarray(mask) > 0
    if values.ndim != 2 or values.shape != binary.shape:
        raise ValueError("A same-shaped 2D image and mask are required")
    if connectivity not in (4, 8):
        raise ValueError("2D connectivity must be 4 or 8")
    row_bounds = _axis_bounds(values.shape[0], grid[0])
    col_bounds = _axis_bounds(values.shape[1], grid[1])
    index_by_grid: dict[tuple[int, int], int] = {}
    features: list[np.ndarray] = []
    positions: list[tuple[float, float]] = []
    for row, (y0, y1) in enumerate(row_bounds):
        for column, (x0, x1) in enumerate(col_bounds):
            patch_mask = binary[y0:y1, x0:x1]
            if not include_empty_nodes and not patch_mask.any():
                continue
            position = ((row + 0.5) / grid[0], (column + 0.5) / grid[1])
            index_by_grid[(row, column)] = len(features)
            features.append(_patch_features(values[y0:y1, x0:x1], patch_mask, position))
            positions.append(position)
    if not features:
        raise ValueError("ROI mask did not activate any graph nodes")

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    spatial: set[tuple[int, int]] = set()
    for coordinate, source in index_by_grid.items():
        for offset in offsets:
            target_coordinate = (coordinate[0] + offset[0], coordinate[1] + offset[1])
            if target_coordinate in index_by_grid:
                spatial.add((source, index_by_grid[target_coordinate]))
        spatial.add((source, source))
    edges = set(spatial)
    _add_knn_edges(np.stack(features), edges, knn_k)
    return _finalise_graph(
        features,
        positions,
        ["image_patch"] * len(features),
        edges,
        spatial,
        {"graph_type": "patch_2d", "grid": list(grid), "connectivity": connectivity, "knn_k": knn_k},
    )


def build_patch_graph_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    grid: tuple[int, int, int] = (4, 4, 4),
    connectivity: int = 6,
    knn_k: int = 3,
    include_empty_nodes: bool = False,
) -> GraphSample:
    """Build a 3D patch graph in normalised z-y-x coordinates."""
    values = np.asarray(volume, dtype=np.float32)
    binary = np.asarray(mask) > 0
    if values.ndim != 3 or values.shape != binary.shape:
        raise ValueError("A same-shaped 3D volume and mask are required")
    if connectivity not in (6, 26):
        raise ValueError("3D connectivity must be 6 or 26")
    bounds = [_axis_bounds(values.shape[axis], grid[axis]) for axis in range(3)]
    index_by_grid: dict[tuple[int, int, int], int] = {}
    features: list[np.ndarray] = []
    positions: list[tuple[float, float, float]] = []
    for z, (z0, z1) in enumerate(bounds[0]):
        for y, (y0, y1) in enumerate(bounds[1]):
            for x, (x0, x1) in enumerate(bounds[2]):
                patch_mask = binary[z0:z1, y0:y1, x0:x1]
                if not include_empty_nodes and not patch_mask.any():
                    continue
                position = ((z + 0.5) / grid[0], (y + 0.5) / grid[1], (x + 0.5) / grid[2])
                index_by_grid[(z, y, x)] = len(features)
                features.append(_patch_features(values[z0:z1, y0:y1, x0:x1], patch_mask, position))
                positions.append(position)
    if not features:
        raise ValueError("ROI mask did not activate any 3D graph nodes")

    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                manhattan = abs(dz) + abs(dy) + abs(dx)
                if manhattan == 0:
                    continue
                if connectivity == 6 and manhattan != 1:
                    continue
                offsets.append((dz, dy, dx))
    spatial: set[tuple[int, int]] = set()
    for coordinate, source in index_by_grid.items():
        for offset in offsets:
            target_coordinate = tuple(coordinate[axis] + offset[axis] for axis in range(3))
            if target_coordinate in index_by_grid:
                spatial.add((source, index_by_grid[target_coordinate]))
        spatial.add((source, source))
    edges = set(spatial)
    _add_knn_edges(np.stack(features), edges, knn_k)
    return _finalise_graph(
        features,
        positions,
        ["volume_patch"] * len(features),
        edges,
        spatial,
        {"graph_type": "patch_3d", "grid": list(grid), "connectivity": connectivity, "knn_k": knn_k},
    )


def build_multimodal_graph(
    roi_embeddings: Mapping[str, np.ndarray],
    radiomic_features: Mapping[str, float],
    clinical_features: Mapping[str, float],
) -> GraphSample:
    """Create a heterogeneous study graph without connecting patients."""
    if not roi_embeddings:
        raise ValueError("At least one ROI embedding is required")
    raw_nodes: list[tuple[str, str, np.ndarray]] = []
    for name, embedding in roi_embeddings.items():
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError(f"ROI embedding {name} is empty")
        raw_nodes.append((name, "roi", vector))
    if radiomic_features:
        raw_nodes.append(("radiomics", "radiomics", np.asarray(list(radiomic_features.values()), dtype=np.float32)))
    if clinical_features:
        raw_nodes.append(("clinical", "clinical", np.asarray(list(clinical_features.values()), dtype=np.float32)))
    dimension = max(vector.size for _, _, vector in raw_nodes)
    features = []
    for _, _, vector in raw_nodes:
        padded = np.zeros(dimension, dtype=np.float32)
        padded[: vector.size] = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(padded)
    positions = [(index / max(1, len(raw_nodes) - 1), 0.0) for index in range(len(raw_nodes))]
    roi_indices = [index for index, (_, node_type, _) in enumerate(raw_nodes) if node_type == "roi"]
    modality_indices = [index for index, (_, node_type, _) in enumerate(raw_nodes) if node_type != "roi"]
    spatial: set[tuple[int, int]] = set()
    edges: set[tuple[int, int]] = {(index, index) for index in range(len(raw_nodes))}
    for left in roi_indices:
        for right in roi_indices:
            edges.add((left, right))
            spatial.add((left, right))
    for modality in modality_indices:
        for roi in roi_indices:
            edges.add((modality, roi))
            edges.add((roi, modality))
    return _finalise_graph(
        features,
        positions,
        [node_type for _, node_type, _ in raw_nodes],
        edges,
        spatial,
        {
            "graph_type": "multimodal_study",
            "node_names": [name for name, _, _ in raw_nodes],
            "radiomic_feature_names": list(radiomic_features),
            "clinical_feature_names": list(clinical_features),
        },
    )

