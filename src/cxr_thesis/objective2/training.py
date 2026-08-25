"""Shared training utilities for the five Objective 2 model families."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import GraphBatch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_logits(
    model: nn.Module,
    batch: dict[str, torch.Tensor] | GraphBatch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, GraphBatch):
        graph_batch = batch.to(device)
        return model(graph_batch), graph_batch.labels
    image = batch["image"].to(device, non_blocking=True)
    clinical = batch["clinical"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    return model(image, clinical), labels


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    total_cases = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        batch_cases = int(
            batch.labels.shape[0]
            if isinstance(batch, GraphBatch)
            else batch["labels"].shape[0]
        )
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits, labels = model_logits(model, batch, device)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * batch_cases
        total_cases += batch_cases
    return total_loss / max(1, total_cases)


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    for batch in loader:
        logits, labels = model_logits(model, batch, device)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    model_name: str,
    label_names: list[str],
    epoch: int,
    validation_macro_auroc: float,
    seed: int,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_state": model.state_dict(),
            "label_names": label_names,
            "epoch": int(epoch),
            "validation_macro_auroc": float(validation_macro_auroc),
            "seed": int(seed),
            "test_evaluated": False,
        },
        target,
    )
    return target
