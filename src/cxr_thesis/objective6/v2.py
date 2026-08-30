"""Deterministic utilities for the Objective 6 English v2 extension."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def canonical_private_patient(value: object) -> str:
    """Return the stable private patient key used only for cohort partitioning."""

    text = str(value).strip()
    if not text or text.casefold() == "nan":
        raise ValueError("Objective 6 v2 requires a nonempty patient identifier")
    return text


def patient_hash(patient: object, *, seed: int) -> str:
    """Hash a private patient identifier without exposing it publicly."""

    key = canonical_private_patient(patient)
    return hashlib.sha256(f"objective6-v2|{seed}|{key}".encode("utf-8")).hexdigest()


def select_development_patients(
    patients: Iterable[object],
    *,
    seed: int,
    fraction: float,
) -> set[str]:
    """Select an exact deterministic fraction of unique development patients.

    Selection is label-, report-, prediction-, and image-blind. The returned
    identifiers are private and must never be written to a public artifact.
    """

    if not 0.0 < fraction < 1.0:
        raise ValueError("Development fraction must be strictly between zero and one")
    unique = sorted({canonical_private_patient(value) for value in patients})
    if len(unique) < 2:
        raise ValueError("At least two unique patients are required")
    count = max(1, min(len(unique) - 1, round(len(unique) * fraction)))
    ranked = sorted(unique, key=lambda value: (patient_hash(value, seed=seed), value))
    return set(ranked[:count])
