"""Deterministic seeding and the frozen seed list for Objective 3 v2.0."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

SEED_BASE = 42
SEED_STRIDE = 1000


def protocol_seeds(count: int) -> list[int]:
    """Return the study's seed list, generated deterministically.

    The protocol fixes the seed list before any run and writes it out in full.
    Seeds are never added after results have been seen.
    """

    if count < 1:
        raise ValueError("At least one seed is required")
    return [SEED_BASE + SEED_STRIDE * index for index in range(count)]


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and (if installed) torch and PennyLane.

    Returns a record of what was seeded, for the results JSON.
    """

    value = int(seed)
    record: dict[str, Any] = {"seed": value, "python": True, "numpy": True}
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    np.random.seed(value)

    try:
        import torch
    except ImportError:
        record["torch"] = False
    else:
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
            record["cuda"] = True
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        record["torch"] = True
        record["torch_version"] = torch.__version__

    try:
        import pennylane
    except ImportError:
        record["pennylane"] = False
    else:
        record["pennylane"] = True
        record["pennylane_version"] = pennylane.version()

    return record


def new_generator(seed: int) -> np.random.Generator:
    """Return an independent NumPy generator for bootstrap or subsampling work."""

    return np.random.default_rng(int(seed))
