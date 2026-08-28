"""Locked DenseNet adaptation helpers for Objective 5."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from cxr_thesis.objective2.models import build_classifier


def _output_linear(model: nn.Module) -> nn.Linear:
    layer = model.classifier[-1]
    if not isinstance(layer, nn.Linear):
        raise TypeError("DenseNet classifier does not end in a linear layer")
    return layer


def initialise_shared_label_densenet(
    checkpoint: dict[str, object],
    target_labels: Sequence[str],
) -> tuple[nn.Module, list[int]]:
    """Build the six-label model and copy matching source output weights."""

    source_labels = [str(value) for value in checkpoint["label_names"]]
    if len(source_labels) != len(set(source_labels)):
        raise ValueError("Source checkpoint label names are not unique")
    missing = [label for label in target_labels if label not in source_labels]
    if missing:
        raise ValueError(f"Target labels are absent from source checkpoint: {missing}")

    model_config = dict(checkpoint.get("model_config", {}))
    clinical_dim = int(model_config.get("clinical_dim", 9))
    dropout = float(model_config.get("dropout", 0.2))
    source = build_classifier(
        "densenet121",
        len(source_labels),
        clinical_dim=clinical_dim,
        pretrained=False,
        dropout=dropout,
    )
    source.load_state_dict(checkpoint["model_state"], strict=True)
    target = build_classifier(
        "densenet121",
        len(target_labels),
        clinical_dim=clinical_dim,
        pretrained=False,
        dropout=dropout,
    )
    target.encoder.load_state_dict(source.encoder.state_dict(), strict=True)
    target.clinical.load_state_dict(source.clinical.state_dict(), strict=True)
    indices = [source_labels.index(label) for label in target_labels]
    source_output = _output_linear(source)
    target_output = _output_linear(target)
    with torch.no_grad():
        target_output.weight.copy_(source_output.weight[indices])
        target_output.bias.copy_(source_output.bias[indices])
    return target, indices


def set_adaptation_phase(model: nn.Module, phase: str) -> None:
    """Apply the preregistered warm-up or final-block fine-tuning policy."""

    if phase not in {"head_warmup", "final_block"}:
        raise ValueError("phase must be head_warmup or final_block")
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.clinical.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if phase == "final_block":
        for parameter in model.encoder.features.denseblock4.parameters():
            parameter.requires_grad = True
        for parameter in model.encoder.features.norm5.parameters():
            parameter.requires_grad = True


def final_block_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.encoder.features.denseblock4.parameters()) + list(
        model.encoder.features.norm5.parameters()
    )


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.clinical.parameters()) + list(model.classifier.parameters())
