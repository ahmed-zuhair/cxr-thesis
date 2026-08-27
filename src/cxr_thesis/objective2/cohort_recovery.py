"""Deterministic, label-blind recovery of locked Objective 2 cohorts."""

from __future__ import annotations

import hashlib
import io
import random
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def _numeric_patient_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def patient_orders(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return the historical candidate patient orderings without using labels."""
    patients = frame["patient_id"].astype(str).drop_duplicates().tolist()
    return {
        "manifest_unique": patients,
        "patient_numeric": sorted(patients, key=_numeric_patient_key),
        "patient_string": sorted(patients),
    }


def greedy_complete_patient_selection(
    frame: pd.DataFrame,
    *,
    ordered_patients: Sequence[str],
    seed: int,
    target_images: int,
    randomizer: str = "default_rng",
) -> list[str]:
    """Select whole patients greedily until an exact image capacity is reached.

    Only ``patient_id`` is inspected. Disease labels, predictions, and risk
    scores cannot influence the selected identities.
    """
    if target_images <= 0:
        raise ValueError("target_images must be positive")
    counts = frame["patient_id"].astype(str).value_counts(sort=False).to_dict()
    ordered = list(map(str, ordered_patients))
    if set(ordered) != set(counts) or len(ordered) != len(counts):
        raise ValueError("ordered_patients must contain every patient exactly once")

    if randomizer == "default_rng":
        shuffled = np.random.default_rng(seed).permutation(ordered).tolist()
    elif randomizer == "random_state":
        shuffled = np.random.RandomState(seed).permutation(ordered).tolist()
    elif randomizer == "python_random":
        shuffled = ordered.copy()
        random.Random(seed).shuffle(shuffled)
    else:
        raise ValueError(f"Unsupported randomizer: {randomizer}")

    selected: list[str] = []
    images = 0
    for patient_id in shuffled:
        patient_images = int(counts[str(patient_id)])
        if images + patient_images <= target_images:
            selected.append(str(patient_id))
            images += patient_images
            if images == target_images:
                break
    if images != target_images:
        raise RuntimeError(
            f"Could not reach exactly {target_images} images; selected {images}"
        )
    return selected


def serialize_cohort(
    full_manifest: pd.DataFrame,
    *,
    selected_patients: Iterable[str],
    row_order: str,
    selection_order: Sequence[str] | None = None,
) -> bytes:
    """Serialize one private cohort using a declared deterministic row order."""
    selected = set(map(str, selected_patients))
    frame = full_manifest[
        full_manifest["patient_id"].astype(str).isin(selected)
    ].copy()
    if row_order == "manifest":
        pass
    elif row_order == "image_id":
        frame = frame.sort_values("image_id", kind="stable")
    elif row_order == "patient_numeric":
        frame["__patient_numeric"] = frame["patient_id"].map(_numeric_patient_key)
        frame = frame.sort_values(["__patient_numeric", "image_id"], kind="stable")
        frame = frame.drop(columns="__patient_numeric")
    elif row_order == "patient_string":
        frame = frame.sort_values(["patient_id", "image_id"], kind="stable")
    elif row_order == "selection":
        if selection_order is None:
            raise ValueError("selection_order is required for selection row order")
        ranks = {str(patient): index for index, patient in enumerate(selection_order)}
        frame["__selection_rank"] = frame["patient_id"].astype(str).map(ranks)
        frame = frame.sort_values(["__selection_rank", "image_id"], kind="stable")
        frame = frame.drop(columns="__selection_rank")
    else:
        raise ValueError(f"Unsupported row_order: {row_order}")
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def select_disjoint_confirmation_patients(
    identity_frame: pd.DataFrame,
    *,
    excluded_patient_ids: Iterable[str],
    split: str,
    seed: int,
    target_images: int,
) -> list[str]:
    """Select a label-blind complete-patient confirmation cohort.

    Patients in ``excluded_patient_ids`` are removed before the deterministic
    numeric-order/default-RNG selection. Only identity and split values are
    inspected; disease labels, predictions, and risk scores are not inputs.
    """
    required = {"patient_id", "split"}
    missing = required - set(identity_frame.columns)
    if missing:
        raise ValueError(f"Identity frame is missing columns: {sorted(missing)}")
    excluded = set(map(str, excluded_patient_ids))
    eligible = identity_frame[
        identity_frame["split"].astype(str).str.lower().eq(split.lower())
        & ~identity_frame["patient_id"].astype(str).isin(excluded)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No eligible confirmation patients remain")
    ordered = patient_orders(eligible)["patient_numeric"]
    selected = greedy_complete_patient_selection(
        eligible,
        ordered_patients=ordered,
        seed=seed,
        target_images=target_images,
        randomizer="default_rng",
    )
    if set(selected) & excluded:
        raise RuntimeError("Confirmation cohort overlaps excluded patients")
    return selected


def recover_exact_cohort_bytes(
    identity_frame: pd.DataFrame,
    full_manifest: pd.DataFrame,
    *,
    split: str,
    seed: int,
    target_images: int,
    expected_patients: int,
    expected_sha256: str,
) -> tuple[bytes, dict[str, object]]:
    """Search documented historical variants and return only an exact match."""
    split_identity = identity_frame[
        identity_frame["split"].astype(str).str.lower() == split.lower()
    ].copy()
    split_manifest = full_manifest[
        full_manifest["split"].astype(str).str.lower() == split.lower()
    ].copy()
    if len(split_identity) != len(split_manifest):
        raise ValueError("Identity and full split frames have different row counts")

    matches: list[tuple[bytes, dict[str, object]]] = []
    attempts: list[dict[str, object]] = []
    for patient_order_name, ordered_patients in patient_orders(split_identity).items():
        for randomizer in ("default_rng", "random_state", "python_random"):
            try:
                selected = greedy_complete_patient_selection(
                    split_identity,
                    ordered_patients=ordered_patients,
                    seed=seed,
                    target_images=target_images,
                    randomizer=randomizer,
                )
            except RuntimeError:
                continue
            patient_count = len(selected)
            for row_order in (
                "manifest",
                "image_id",
                "patient_numeric",
                "patient_string",
                "selection",
            ):
                payload = serialize_cohort(
                    split_manifest,
                    selected_patients=selected,
                    row_order=row_order,
                    selection_order=selected,
                )
                digest = sha256_bytes(payload)
                record: dict[str, object] = {
                    "patient_order": patient_order_name,
                    "randomizer": randomizer,
                    "row_order": row_order,
                    "images": target_images,
                    "patients": patient_count,
                    "sha256": digest,
                }
                attempts.append(record)
                if digest == expected_sha256 and patient_count == expected_patients:
                    matches.append((payload, record))

    if not matches:
        identity_matches = [
            item
            for item in attempts
            if int(item["patients"]) == expected_patients
        ]
        raise RuntimeError(
            "No deterministic cohort variant reproduced the protected SHA-256; "
            f"{len(identity_matches)} variants matched the patient count"
        )
    unique_payloads = {payload for payload, _ in matches}
    if len(unique_payloads) != 1:
        raise RuntimeError("Protected SHA-256 unexpectedly matched different payloads")
    payload = next(iter(unique_payloads))
    matching_variants = [record for candidate, record in matches if candidate == payload]
    return payload, {
        "matching_variants": matching_variants,
        "attempted_variants": len(attempts),
        "selection_used_labels": False,
        "selection_used_predictions": False,
        "selection_used_risk_scores": False,
    }
