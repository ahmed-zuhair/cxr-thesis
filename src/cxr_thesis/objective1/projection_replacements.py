"""Reversible finalization of prediction-blind projection replacements."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


BASE_WORKLIST_COLUMNS = [
    "candidate_code",
    "cohort_role",
    "view",
    "sex",
    "finding_group",
    "selection_basis",
    "image_filename",
    "preannotation_filename",
    "required_output_mask",
]

AUDIT_COLUMNS = [
    "candidate_code",
    "cohort_role",
    "projection_decision",
    "auditor",
    "updated_utc",
    "note",
]

DEVELOPMENT_ROLES = ("adaptation_train", "target_validation")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_unique_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    if frame["candidate_code"].astype(str).duplicated().any():
        raise ValueError(f"{path.name} contains duplicate candidate codes")
    return frame


def _count_png(path: Path) -> int:
    return sum(1 for candidate in path.glob("*.png") if candidate.is_file())


def finalize_projection_replacements(
    cohort_workspace: str | Path,
    reserve_workspace: str | Path,
    transaction_root: str | Path,
) -> dict[str, object]:
    """Apply one rank-first eligible reserve to each rejected development slot.

    The original locked test is hashed before and after the transaction. Original
    rejected files remain on disk as unreferenced evidence, while the previous
    worklists and audits are copied into the private transaction directory.
    """

    cohort_root = Path(cohort_workspace).resolve()
    reserve_root = Path(reserve_workspace).resolve()
    transaction = Path(transaction_root).resolve()
    if transaction.exists():
        raise FileExistsError(f"Transaction output already exists: {transaction}")
    if not cohort_root.is_dir():
        raise FileNotFoundError(cohort_root)
    if not reserve_root.is_dir():
        raise FileNotFoundError(reserve_root)

    identity_path = reserve_root / "replacement_identity_private.csv"
    if not identity_path.is_file():
        raise FileNotFoundError(identity_path)
    identity = pd.read_csv(identity_path, keep_default_na=False)
    identity_required = {
        "replacement_code",
        "candidate_code",
        "cohort_role",
        "replacement_slot",
        "reserve_rank",
    }
    identity_missing = sorted(identity_required - set(identity.columns))
    if identity_missing:
        raise ValueError(
            f"Replacement identity is missing columns: {identity_missing}"
        )
    if identity.empty:
        raise ValueError("Replacement identity is empty")
    identity = identity.rename(columns={"candidate_code": "replaces_candidate_code"})
    if identity["replacement_code"].astype(str).duplicated().any():
        raise ValueError("Replacement identity contains duplicate replacement codes")

    recovery_summary_path = reserve_root / "replacement_recovery_summary_private.json"
    if not recovery_summary_path.is_file():
        raise FileNotFoundError(recovery_summary_path)
    recovery_summary = json.loads(recovery_summary_path.read_text(encoding="utf-8"))
    if recovery_summary.get("official_nih_test_used") is not False:
        raise RuntimeError("Reserve recovery did not prove NIH-test exclusion")
    if recovery_summary.get("locked_target_test_modified") is not False:
        raise RuntimeError("Reserve recovery reports a locked-test modification")
    if recovery_summary.get("replacement_selection_uses_predictions") is not False:
        raise RuntimeError("Reserve selection was not prediction-blind")

    locked_root = cohort_root / "locked_target_test"
    locked_protected = [
        locked_root / "annotation_worklist.csv",
        locked_root / "projection_audit.csv",
    ]
    for path in locked_protected:
        if not path.is_file():
            raise FileNotFoundError(path)
    locked_hashes_before = {path.name: sha256_file(path) for path in locked_protected}
    if _count_png(locked_root / "annotations") != 0:
        raise RuntimeError("Locked target test already contains real annotations")

    plans: list[dict[str, object]] = []
    for role in DEVELOPMENT_ROLES:
        cohort_role_root = cohort_root / role
        reserve_role_root = reserve_root / role
        worklist_path = cohort_role_root / "annotation_worklist.csv"
        audit_path = cohort_role_root / "projection_audit.csv"
        reserve_worklist_path = reserve_role_root / "annotation_worklist.csv"
        reserve_audit_path = reserve_role_root / "projection_audit.csv"

        worklist = _read_unique_csv(worklist_path, set(BASE_WORKLIST_COLUMNS))
        audit = _read_unique_csv(audit_path, set(AUDIT_COLUMNS))
        reserve_worklist = _read_unique_csv(
            reserve_worklist_path,
            set(BASE_WORKLIST_COLUMNS)
            | {"replacement_slot", "reserve_rank", "replacement_selection_basis"},
        )
        reserve_audit = _read_unique_csv(reserve_audit_path, set(AUDIT_COLUMNS))
        if set(worklist["cohort_role"].astype(str)) != {role}:
            raise ValueError(f"Original worklist role mismatch for {role}")
        if set(reserve_worklist["cohort_role"].astype(str)) != {role}:
            raise ValueError(f"Reserve worklist role mismatch for {role}")
        if len(reserve_worklist) != 5:
            raise ValueError(f"Expected five reserves for {role}")
        if reserve_worklist["replacement_slot"].astype(str).nunique() != 1:
            raise ValueError(f"Expected one replacement slot for {role}")
        reserve_ranks = set(
            pd.to_numeric(reserve_worklist["reserve_rank"], errors="raise").astype(int)
        )
        if reserve_ranks != {1, 2, 3, 4, 5}:
            raise ValueError(f"Reserve ranks are incomplete for {role}")
        if set(reserve_worklist["replacement_selection_basis"].astype(str)) != {
            "projection_blind_hash"
        }:
            raise ValueError(f"Replacement selection basis changed for {role}")
        if set(reserve_audit["candidate_code"]) != set(
            reserve_worklist["candidate_code"]
        ):
            raise ValueError(f"Reserve audit is incomplete for {role}")
        rejected = audit[audit["projection_decision"] != "eligible_frontal"]
        if len(rejected) != 1:
            raise ValueError(f"Expected exactly one rejected original case for {role}")
        if _count_png(cohort_role_root / "annotations") != 0:
            raise RuntimeError(f"{role} already contains real annotations")

        merged = reserve_worklist.merge(
            reserve_audit[["candidate_code", "projection_decision", "auditor", "updated_utc", "note"]],
            on="candidate_code",
            how="left",
            validate="one_to_one",
        ).merge(
            identity[
                [
                    "replacement_code",
                    "replaces_candidate_code",
                    "cohort_role",
                    "replacement_slot",
                    "reserve_rank",
                ]
            ],
            left_on="candidate_code",
            right_on="replacement_code",
            how="left",
            suffixes=("", "_identity"),
            validate="one_to_one",
        )
        if merged["replaces_candidate_code"].eq("").any():
            raise ValueError(f"Reserve identity is incomplete for {role}")
        if set(merged["cohort_role_identity"].astype(str)) != {role}:
            raise ValueError(f"Reserve identity role mismatch for {role}")
        if not (
            merged["replacement_slot"].astype(str)
            == merged["replacement_slot_identity"].astype(str)
        ).all():
            raise ValueError(f"Reserve identity slot mismatch for {role}")
        if not (
            pd.to_numeric(merged["reserve_rank"], errors="raise").astype(int)
            == pd.to_numeric(merged["reserve_rank_identity"], errors="raise").astype(int)
        ).all():
            raise ValueError(f"Reserve identity rank mismatch for {role}")
        if merged["replaces_candidate_code"].nunique() != 1:
            raise ValueError(f"Reserve role {role} refers to multiple original cases")
        original_code = str(rejected.iloc[0]["candidate_code"])
        if str(merged.iloc[0]["replaces_candidate_code"]) != original_code:
            raise ValueError(f"Reserve identity does not match rejected {role} case")
        eligible = merged[merged["projection_decision"] == "eligible_frontal"].copy()
        if eligible.empty:
            raise ValueError(f"No eligible replacement reserve exists for {role}")
        eligible["reserve_rank"] = pd.to_numeric(eligible["reserve_rank"], errors="raise")
        chosen = eligible.sort_values(
            ["reserve_rank", "candidate_code"], kind="stable"
        ).iloc[0]
        original = worklist[worklist["candidate_code"] == original_code].iloc[0]
        for column in ("view", "sex", "finding_group", "selection_basis"):
            if str(chosen[column]) != str(original[column]):
                raise ValueError(f"Replacement {role} stratum mismatch in {column}")

        source_image = reserve_role_root / str(chosen["image_filename"])
        source_mask = reserve_role_root / str(chosen["preannotation_filename"])
        target_image = cohort_role_root / str(chosen["image_filename"])
        target_mask = cohort_role_root / str(chosen["preannotation_filename"])
        target_annotation = cohort_role_root / str(chosen["required_output_mask"])
        for source in (source_image, source_mask):
            if not source.is_file():
                raise FileNotFoundError(source)
        for target in (target_image, target_mask, target_annotation):
            if target.exists():
                raise FileExistsError(target)
        with Image.open(source_image) as image, Image.open(source_mask) as mask:
            if image.size != mask.size:
                raise RuntimeError(f"Replacement image/mask shape mismatch for {role}")

        plans.append(
            {
                "role": role,
                "worklist_path": worklist_path,
                "audit_path": audit_path,
                "reserve_audit_path": reserve_audit_path,
                "worklist": worklist,
                "audit": audit,
                "original_code": original_code,
                "chosen": chosen,
                "source_image": source_image,
                "source_mask": source_mask,
                "target_image": target_image,
                "target_mask": target_mask,
            }
        )

    transaction.mkdir(parents=True)
    backup_root = transaction / "before_replacement"
    backup_root.mkdir()
    shutil.copy2(identity_path, transaction / identity_path.name)
    copied_targets: list[Path] = []
    applied_rows: list[dict[str, object]] = []
    try:
        for plan in plans:
            role = str(plan["role"])
            role_backup = backup_root / role
            role_backup.mkdir()
            worklist_path = Path(plan["worklist_path"])
            audit_path = Path(plan["audit_path"])
            shutil.copy2(worklist_path, role_backup / worklist_path.name)
            shutil.copy2(audit_path, role_backup / audit_path.name)
            shutil.copy2(
                Path(plan["reserve_audit_path"]),
                role_backup / "replacement_projection_audit.csv",
            )
            target_image = Path(plan["target_image"])
            target_mask = Path(plan["target_mask"])
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(plan["source_image"]), target_image)
            copied_targets.append(target_image)
            shutil.copy2(Path(plan["source_mask"]), target_mask)
            copied_targets.append(target_mask)

            worklist = pd.DataFrame(plan["worklist"])
            original_code = str(plan["original_code"])
            chosen = plan["chosen"]
            replacement_row = {
                column: str(chosen[column]) for column in BASE_WORKLIST_COLUMNS
            }
            updated_worklist = pd.concat(
                [
                    worklist[worklist["candidate_code"] != original_code],
                    pd.DataFrame([replacement_row]),
                ],
                ignore_index=True,
            ).sort_values("candidate_code", kind="stable")
            if len(updated_worklist) != len(worklist):
                raise RuntimeError(f"Worklist count changed for {role}")

            audit = pd.DataFrame(plan["audit"])
            replacement_audit_row = {
                "candidate_code": str(chosen["candidate_code"]),
                "cohort_role": role,
                "projection_decision": "eligible_frontal",
                "auditor": str(chosen["auditor"]),
                "updated_utc": str(chosen["updated_utc"]),
                "note": str(chosen["note"]),
            }
            updated_audit = pd.concat(
                [
                    audit[audit["candidate_code"] != original_code],
                    pd.DataFrame([replacement_audit_row]),
                ],
                ignore_index=True,
            ).sort_values("candidate_code", kind="stable")
            if len(updated_audit) != len(audit):
                raise RuntimeError(f"Projection-audit count changed for {role}")
            if set(updated_audit["projection_decision"]) != {"eligible_frontal"}:
                raise RuntimeError(f"Non-frontal decisions remain for {role}")
            _atomic_csv(updated_worklist[BASE_WORKLIST_COLUMNS], worklist_path)
            _atomic_csv(updated_audit[AUDIT_COLUMNS], audit_path)
            applied_rows.append(
                {
                    "cohort_role": role,
                    "replaced_candidate_code": original_code,
                    "replacement_candidate_code": str(chosen["candidate_code"]),
                    "replacement_slot": str(chosen["replacement_slot"]),
                    "reserve_rank": int(chosen["reserve_rank"]),
                    "replacement_selection_basis": str(
                        chosen["replacement_selection_basis"]
                    ),
                    "replacement_image_sha256": sha256_file(target_image),
                    "replacement_preannotation_sha256": sha256_file(target_mask),
                    "replacement_image_path": str(target_image),
                    "replacement_preannotation_path": str(target_mask),
                }
            )
    except Exception:
        for plan in plans:
            role_backup = backup_root / str(plan["role"])
            worklist_path = Path(plan["worklist_path"])
            audit_path = Path(plan["audit_path"])
            if (role_backup / worklist_path.name).is_file():
                shutil.copy2(role_backup / worklist_path.name, worklist_path)
            if (role_backup / audit_path.name).is_file():
                shutil.copy2(role_backup / audit_path.name, audit_path)
        for target in copied_targets:
            target.unlink(missing_ok=True)
        raise

    locked_hashes_after = {path.name: sha256_file(path) for path in locked_protected}
    if locked_hashes_after != locked_hashes_before:
        raise RuntimeError("Locked target test changed during replacement finalization")
    role_counts = {}
    role_eligible_counts = {}
    for role in (*DEVELOPMENT_ROLES, "locked_target_test"):
        role_root = cohort_root / role
        role_counts[role] = int(len(pd.read_csv(role_root / "annotation_worklist.csv")))
        role_audit = pd.read_csv(role_root / "projection_audit.csv")
        role_eligible_counts[role] = int(
            (role_audit["projection_decision"] == "eligible_frontal").sum()
        )

    applied_path = transaction / "applied_projection_replacements_private.csv"
    pd.DataFrame(applied_rows).to_csv(applied_path, index=False)

    summary = {
        "artifact": "Private finalized Objective 1 projection replacements",
        "selection_rule": "lowest reserve_rank among eligible_frontal reserves",
        "replacement_slots": len(plans),
        "selected_replacements": len(plans),
        "role_counts": role_counts,
        "eligible_frontal_counts": role_eligible_counts,
        "locked_target_test_hashes_before": locked_hashes_before,
        "locked_target_test_hashes_after": locked_hashes_after,
        "locked_target_test_modified": False,
        "official_nih_test_used_for_replacement": False,
        "real_annotations_present": False,
        "reserve_recovery_summary_sha256": sha256_file(recovery_summary_path),
        "applied_replacement_record": str(applied_path),
        "rollback_directory": str(backup_root),
        "private_publication_allowed": False,
    }
    summary_path = transaction / "projection_replacement_finalization_private.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
