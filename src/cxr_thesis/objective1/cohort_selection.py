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


def select_projection_replacement_reserves(
    mapping: pd.DataFrame,
    ranked_audit: pd.DataFrame,
    current_cohort: pd.DataFrame,
    rejected_cases: pd.DataFrame,
    *,
    reserves_per_slot: int = 5,
) -> pd.DataFrame:
    """Select deterministic same-stratum reserves for rejected development cases.

    Adaptation and validation replacements retain the original split, stratum,
    and selection basis. The locked target test is deliberately unsupported so
    its membership cannot be modified by prediction-aware replacement logic.
    """
    if reserves_per_slot < 1:
        raise ValueError("reserves_per_slot must be positive")
    normalised = _normalise_mapping(mapping)
    cohort_required = {
        "candidate_code",
        "patient_id",
        "image_id",
        "cohort_role",
        "cohort_stratum",
        "selection_basis",
        "split",
    }
    missing_cohort = sorted(cohort_required - set(current_cohort.columns))
    if missing_cohort:
        raise ValueError(f"Current cohort is missing columns: {missing_cohort}")
    request_required = {"candidate_code", "cohort_role", "projection_decision"}
    missing_request = sorted(request_required - set(rejected_cases.columns))
    if missing_request:
        raise ValueError(f"Replacement request is missing columns: {missing_request}")
    if rejected_cases.empty:
        raise ValueError("Replacement request is empty")
    if rejected_cases["candidate_code"].duplicated().any():
        raise ValueError("Replacement request contains duplicate candidate codes")
    allowed_roles = {"adaptation_train", "target_validation"}
    requested_roles = set(rejected_cases["cohort_role"].astype(str))
    if not requested_roles.issubset(allowed_roles):
        raise ValueError("Only adaptation or validation cases may be replaced")
    if (rejected_cases["projection_decision"] == "eligible_frontal").any():
        raise ValueError("Eligible frontal cases must not be replaced")

    cohort = current_cohort.copy()
    cohort["candidate_code"] = cohort["candidate_code"].astype(str)
    request = rejected_cases.copy()
    request["candidate_code"] = request["candidate_code"].astype(str)
    request["cohort_role"] = request["cohort_role"].astype(str)
    originals = request.merge(
        cohort,
        on=["candidate_code", "cohort_role"],
        how="left",
        validate="one_to_one",
        suffixes=("_request", ""),
    )
    if originals["image_id"].isna().any():
        raise ValueError("A requested case is absent from the current cohort")

    required_risk = {
        "image_id",
        "active_qc_priority_score",
        "active_qc_risk_score",
        "mask_path",
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
    pool = normalised.merge(
        ranked_audit[risk_columns], on="image_id", how="left", validate="one_to_one"
    )
    if pool["active_qc_risk_score"].isna().any():
        raise ValueError("Some replacement candidates are missing active-QC scores")

    used_images = set(cohort["image_id"].astype(str))
    used_patients = set(cohort["patient_id"].astype(str))
    selected: list[pd.DataFrame] = []
    ordered_originals = originals.sort_values(
        ["cohort_role", "candidate_code"], kind="stable"
    ).reset_index(drop=True)
    ranked_lookup = ranked_audit.set_index(ranked_audit["image_id"].astype(str))

    for slot_index, original in ordered_originals.iterrows():
        role = str(original["cohort_role"])
        stratum = str(original["cohort_stratum"])
        split = str(original["split"]).lower()
        basis = str(original["selection_basis"])
        group = pool[
            (pool["split"] == split)
            & (pool["cohort_stratum"] == stratum)
            & (~pool["image_id"].astype(str).isin(used_images))
            & (~pool["patient_id"].astype(str).isin(used_patients))
        ].copy()
        if basis == "active_qc_high_risk":
            group = group.sort_values(
                ["active_qc_priority_score", "active_qc_risk_score", "candidate_code"],
                ascending=[False, False, True],
                kind="stable",
            )
        elif basis == "active_qc_representative":
            original_risk = float(
                ranked_lookup.loc[str(original["image_id"]), "active_qc_risk_score"]
            )
            group["_risk_distance"] = (
                group["active_qc_risk_score"].astype(float) - original_risk
            ).abs()
            group = group.sort_values(
                ["_risk_distance", "active_qc_risk_score", "candidate_code"],
                kind="stable",
            ).drop(columns="_risk_distance")
        else:
            raise ValueError(f"Unsupported replacement selection basis: {basis!r}")
        if len(group) < reserves_per_slot:
            raise ValueError(
                f"Replacement slot {slot_index + 1} has only {len(group)} eligible pool "
                f"rows; {reserves_per_slot} reserves are required"
            )
        chosen = group.head(reserves_per_slot).copy()
        chosen["replacement_slot"] = f"RPL-{slot_index + 1:02d}"
        chosen["reserve_rank"] = np.arange(1, reserves_per_slot + 1)
        chosen["cohort_role"] = role
        chosen["cohort_stratum"] = stratum
        chosen["selection_basis"] = basis
        chosen["rejected_projection_decision"] = str(
            original["projection_decision"]
        )
        selected.append(chosen)
        used_images.update(chosen["image_id"].astype(str))
        used_patients.update(chosen["patient_id"].astype(str))

    result = pd.concat(selected, ignore_index=True)
    if result["image_id"].duplicated().any() or result["patient_id"].duplicated().any():
        raise RuntimeError("Replacement reserves are not patient/image disjoint")
    return result.sort_values(
        ["replacement_slot", "reserve_rank"], kind="stable"
    ).reset_index(drop=True)
