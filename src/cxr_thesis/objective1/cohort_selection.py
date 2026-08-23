"""Deterministic selection of a private ROI annotation cohort."""

from __future__ import annotations

import hashlib
from itertools import product

import numpy as np
import pandas as pd


COHORT_STRATA = tuple(
    f"{view}|{sex}|{finding}"
    for view, sex, finding in product(
        ("PA", "AP"),
        ("F", "M"),
        ("no_finding", "abnormal"),
    )
)


def _stable_key(seed: int, purpose: str, patient_id: object, image_id: object) -> str:
    value = f"{seed}|{purpose}|{patient_id}|{image_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_code",
        "patient_id",
        "image_id",
        "image_path",
        "split",
        "view_group",
        "sex_group",
        "finding_group",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate mapping is missing columns: {missing}")

    result = frame.copy()
    result["split"] = result["split"].astype(str).str.strip().str.lower()
    result["view_group"] = result["view_group"].astype(str).str.strip().str.upper()
    result["sex_group"] = result["sex_group"].astype(str).str.strip().str.upper()
    result["finding_group"] = (
        result["finding_group"].astype(str).str.strip().str.lower()
    )
    result["cohort_stratum"] = (
        result["view_group"]
        + "|"
        + result["sex_group"]
        + "|"
        + result["finding_group"]
    )

    if result["patient_id"].duplicated().any():
        raise ValueError("Candidate mapping must contain one row per patient")
    if result["image_id"].duplicated().any():
        raise ValueError("Candidate mapping contains duplicate image IDs")
    if result["candidate_code"].duplicated().any():
        raise ValueError("Candidate mapping contains duplicate candidate codes")
    if not set(result["split"]).issubset({"train", "val"}):
        raise ValueError("Only NIH development train/validation cases are allowed")
    unknown = sorted(set(result["cohort_stratum"]) - set(COHORT_STRATA))
    if unknown:
        raise ValueError(f"Unsupported cohort strata: {unknown}")
    return result


def _select_blind_locked_test(
    mapping: pd.DataFrame,
    *,
    seed: int,
    cases_per_stratum: int,
) -> pd.DataFrame:
    """Select the locked test using identifiers only, before risk is available."""
    pool = mapping[mapping["split"] == "val"].copy()
    selected: list[pd.DataFrame] = []
    for stratum in COHORT_STRATA:
        group = pool[pool["cohort_stratum"] == stratum].copy()
        if len(group) < cases_per_stratum:
            raise ValueError(
                f"Locked-test stratum {stratum} has {len(group)} cases; "
                f"{cases_per_stratum} are required"
            )
        group["_blind_key"] = [
            _stable_key(seed, "locked-target-test", patient, image)
            for patient, image in zip(group["patient_id"], group["image_id"])
        ]
        chosen = group.sort_values(
            ["_blind_key", "candidate_code"], kind="stable"
        ).head(cases_per_stratum)
        selected.append(chosen.drop(columns="_blind_key"))

    result = pd.concat(selected, ignore_index=True)
    result["cohort_role"] = "locked_target_test"
    result["selection_basis"] = "prediction_blind_hash"
    result["selection_order"] = np.arange(1, len(result) + 1)
    return result


def _representative_indices(length: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=int)
    if length < count:
        raise ValueError(f"Cannot choose {count} representatives from {length} rows")
    if count == 1:
        return np.array([(length - 1) // 2], dtype=int)
    return np.rint(np.linspace(0, length - 1, count)).astype(int)


def _select_active_qc_group(
    group: pd.DataFrame,
    *,
    total: int,
    high_risk: int,
    role: str,
) -> pd.DataFrame:
    if len(group) < total:
        raise ValueError(f"{role} group has {len(group)} cases; {total} are required")
    if not 0 <= high_risk <= total:
        raise ValueError("high_risk must be between zero and total")

    ordered_high = group.sort_values(
        ["active_qc_priority_score", "active_qc_risk_score", "candidate_code"],
        ascending=[False, False, True],
        kind="stable",
    )
    high = ordered_high.head(high_risk).copy()
    high["selection_basis"] = "active_qc_high_risk"

    remaining = group[~group["image_id"].isin(high["image_id"])].sort_values(
        ["active_qc_risk_score", "candidate_code"], kind="stable"
    )
    representative_count = total - high_risk
    indices = _representative_indices(len(remaining), representative_count)
    representative = remaining.iloc[indices].copy()
    representative["selection_basis"] = "active_qc_representative"

    selected = pd.concat([high, representative], ignore_index=True)
    selected["cohort_role"] = role
    return selected


def select_roi_annotation_cohort(
    mapping: pd.DataFrame,
    ranked_audit: pd.DataFrame,
    *,
    seed: int = 42,
    adaptation_cases_per_stratum: int = 15,
    adaptation_high_risk_per_stratum: int = 7,
    validation_cases_per_stratum: int = 5,
    validation_high_risk_per_stratum: int = 2,
    locked_test_cases_per_stratum: int = 5,
) -> dict[str, pd.DataFrame]:
    """Lock a balanced 120/40/40 train/validation/test annotation cohort.

    The locked target test is selected from validation identifiers before the
    ranked audit is joined. Consequently its membership cannot depend on model
    predictions, uncertainty proxies, ROI shape, or automatic QC results.
    """
    normalised = _normalise_mapping(mapping)

    # This must remain before the ranked-audit validation and merge.
    locked_test = _select_blind_locked_test(
        normalised,
        seed=seed,
        cases_per_stratum=locked_test_cases_per_stratum,
    )
    locked_ids = set(locked_test["image_id"].astype(str))

    required_risk = {
        "image_id",
        "active_qc_priority_score",
        "active_qc_risk_score",
    }
    missing_risk = sorted(required_risk - set(ranked_audit.columns))
    if missing_risk:
        raise ValueError(f"Ranked audit is missing columns: {missing_risk}")
    if ranked_audit["image_id"].duplicated().any():
        raise ValueError("Ranked audit contains duplicate image IDs")

    risk_columns = [
        column
        for column in ranked_audit.columns
        if column == "image_id" or column not in normalised.columns
    ]
    active_pool = normalised[~normalised["image_id"].astype(str).isin(locked_ids)].merge(
        ranked_audit[risk_columns], on="image_id", how="left", validate="one_to_one"
    )
    if active_pool["active_qc_priority_score"].isna().any():
        raise ValueError("Some non-test candidates are missing active-QC scores")

    selected_adaptation: list[pd.DataFrame] = []
    selected_validation: list[pd.DataFrame] = []
    for stratum in COHORT_STRATA:
        train_group = active_pool[
            (active_pool["split"] == "train")
            & (active_pool["cohort_stratum"] == stratum)
        ]
        selected_adaptation.append(
            _select_active_qc_group(
                train_group,
                total=adaptation_cases_per_stratum,
                high_risk=adaptation_high_risk_per_stratum,
                role="adaptation_train",
            )
        )

        validation_group = active_pool[
            (active_pool["split"] == "val")
            & (active_pool["cohort_stratum"] == stratum)
        ]
        selected_validation.append(
            _select_active_qc_group(
                validation_group,
                total=validation_cases_per_stratum,
                high_risk=validation_high_risk_per_stratum,
                role="target_validation",
            )
        )

    adaptation = pd.concat(selected_adaptation, ignore_index=True)
    validation = pd.concat(selected_validation, ignore_index=True)
    adaptation["selection_order"] = np.arange(1, len(adaptation) + 1)
    validation["selection_order"] = np.arange(1, len(validation) + 1)

    # Locked-test outputs intentionally retain mapping columns only. In
    # particular, they contain no predicted mask path or active-QC metric.
    locked_columns = list(normalised.columns) + [
        "cohort_role",
        "selection_basis",
        "selection_order",
    ]
    locked_test = locked_test.reindex(columns=locked_columns)

    roles = {
        "adaptation_train": adaptation.reset_index(drop=True),
        "target_validation": validation.reset_index(drop=True),
        "locked_target_test": locked_test.reset_index(drop=True),
    }
    combined_ids: list[str] = []
    combined_patients: list[str] = []
    for frame in roles.values():
        combined_ids.extend(frame["image_id"].astype(str))
        combined_patients.extend(frame["patient_id"].astype(str))
    if len(combined_ids) != len(set(combined_ids)):
        raise RuntimeError("Image overlap exists between annotation roles")
    if len(combined_patients) != len(set(combined_patients)):
        raise RuntimeError("Patient overlap exists between annotation roles")

    master = pd.concat(roles.values(), ignore_index=True, sort=False)
    roles["master"] = master
    return roles
