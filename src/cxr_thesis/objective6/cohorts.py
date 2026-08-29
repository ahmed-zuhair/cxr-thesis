"""Deterministic patient-level cohort utilities for Objective 6."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd


FRONTAL_PROJECTIONS = ("PA", "AP", "AP_horizontal")
PROJECTION_RANK = {name: index for index, name in enumerate(FRONTAL_PROJECTIONS)}


def canonical_patient_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return ""
    digits = re.findall(r"\d+", text)
    return str(int(digits[-1])) if digits else text.casefold()


def derive_padchest_age(
    study_date: pd.Series, patient_birth: pd.Series
) -> pd.Series:
    date = pd.to_numeric(study_date, errors="coerce")
    birth = pd.to_numeric(patient_birth, errors="coerce")
    age = np.floor(date / 10000.0) - birth
    return age.where(age.between(0, 120))


def patient_partition(patient_id: object, *, seed: int = 42) -> str:
    """Assign a patient to a 70/15/15 split without labels or report content."""

    patient = canonical_patient_id(patient_id)
    if not patient:
        raise ValueError("Patient identifier is empty")
    digest = hashlib.sha256(f"objective6|{seed}|{patient}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "val"
    return "test"


def private_case_code(patient_id: object, study_id: object, *, seed: int = 42) -> str:
    patient = canonical_patient_id(patient_id)
    digest = hashlib.sha256(
        f"objective6-case|{seed}|{patient}|{study_id}".encode("utf-8")
    ).hexdigest()
    return f"O6-{digest[:16].upper()}"
