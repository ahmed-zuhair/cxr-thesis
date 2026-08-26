"""Shared training utilities for the five Objective 2 model families."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

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
    temporary = target.with_name(f".{target.name}.tmp")
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
        temporary,
    )
    os.replace(temporary, target)
    return target


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG used by Objective 2 training."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state after model and loader construction during resume."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """Move optimizer tensors loaded from a CPU checkpoint to the active device."""

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter_state in optimizer.state.values():
        for key, value in list(parameter_state.items()):
            parameter_state[key] = move(value)


def save_training_state(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    data_loader_generator: torch.Generator,
    epoch_completed: int,
    best_auroc: float,
    stale_epochs: int,
    history: list[dict[str, Any]],
    signature: dict[str, Any],
    best_checkpoint_sha256: str | None,
    resume_count: int,
) -> Path:
    """Atomically save a complete, test-blind epoch-boundary recovery state."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "data_loader_generator_state": data_loader_generator.get_state(),
            "rng_state": capture_rng_state(),
            "epoch_completed": int(epoch_completed),
            "best_auroc": float(best_auroc),
            "stale_epochs": int(stale_epochs),
            "history": history,
            "signature": signature,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "resume_count": int(resume_count),
            "test_evaluated": False,
        },
        temporary,
    )
    os.replace(temporary, target)
    return target
