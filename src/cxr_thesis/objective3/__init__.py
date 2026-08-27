"""Objective 3 hybrid quantum-enhanced graph classification."""

from .embeddings import load_embedding_shard, save_embedding_shard
from .models import (
    ClassicalMatchedBottleneck,
    ClassicalReuploadingBottleneck,
    EnhancedHybridGraphHead,
    HybridGraphHead,
    QuantumBottleneck,
    QuantumReuploadingBottleneck,
    bottleneck_parameter_count,
)
from .training import (
    apply_standardizer,
    fit_standardizer,
    initialize_shared_layers,
    labels_from_manifest,
    load_embedding_corpus,
    positive_weights,
    shared_layer_state,
)

__all__ = [
    "ClassicalMatchedBottleneck",
    "ClassicalReuploadingBottleneck",
    "EnhancedHybridGraphHead",
    "HybridGraphHead",
    "QuantumBottleneck",
    "QuantumReuploadingBottleneck",
    "apply_standardizer",
    "bottleneck_parameter_count",
    "fit_standardizer",
    "initialize_shared_layers",
    "labels_from_manifest",
    "load_embedding_corpus",
    "load_embedding_shard",
    "positive_weights",
    "save_embedding_shard",
    "shared_layer_state",
]
