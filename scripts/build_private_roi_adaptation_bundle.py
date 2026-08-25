"""Create a de-identified private bundle for NIH ROI adaptation on Kaggle.

The bundle intentionally excludes original candidate codes, projection audits,
review notes, and reviewer provenance. It still contains medical images and
annotation masks, so it must remain private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


ROLE_SPEC = {
    "adaptation_train": ("train", "ADAPT", 120),
    "target_validation": ("val", "VALID", 40),
    "locked_target_test": ("test", "LOCKED", 40),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a private de-identified ROI adaptation bundle"
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--locked-test-qc", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    output = Path(args.output_dir).resolve()
    archive = Path(f"{output}.zip")
    checksum = Path(f"{archive}.sha256")
    if output.exists() or archive.exists() or checksum.exists():
        raise FileExistsError("Private bundle output already exists")

    locked_qc = pd.read_csv(args.locked_test_qc, keep_default_na=False)
    if locked_qc["candidate_code"].astype(str).duplicated().any():
        raise ValueError("Locked-test QC contains duplicate candidate codes")
    locked_qc = locked_qc.set_index(locked_qc["candidate_code"].astype(str))

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    rows: list[dict[str, object]] = []
    try:
        for role, (split, prefix, expected) in ROLE_SPEC.items():
            role_root = workspace / role
            worklist = pd.read_csv(
                role_root / "annotation_worklist.csv", keep_default_na=False
            )
            progress = pd.read_csv(
                role_root / "annotation_progress.csv", keep_default_na=False
            )
            if len(worklist) != expected or len(progress) != expected:
                raise ValueError(f"{role} is not complete: expected {expected} cases")
            if worklist["candidate_code"].astype(str).duplicated().any():
                raise ValueError(f"{role} worklist contains duplicates")
            progress = progress.set_index(progress["candidate_code"].astype(str))

            for index, row in enumerate(
                worklist.sort_values("candidate_code", kind="stable").to_dict(
                    orient="records"
                ),
                start=1,
            ):
                original_code = str(row["candidate_code"])
                experiment_id = f"{prefix}-{index:04d}"
                image_source = role_root / str(row["image_filename"])
                mask_source = role_root / str(row["required_output_mask"])
                if not image_source.is_file() or not mask_source.is_file():
                    raise FileNotFoundError(f"A {role} image or mask is missing")
                image_relative = Path("data") / "images" / split / f"{experiment_id}.png"
                mask_relative = Path("data") / "masks" / split / f"{experiment_id}.png"
                image_target = temporary / image_relative
                mask_target = temporary / mask_relative
                image_target.parent.mkdir(parents=True, exist_ok=True)
                mask_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_source, image_target)
                shutil.copy2(mask_source, mask_target)

                progress_status = str(progress.at[original_code, "status"])
                if role == "locked_target_test":
                    if original_code not in locked_qc.index:
                        raise ValueError("Locked-test QC does not match the worklist")
                    requires_review = str(
                        locked_qc.at[original_code, "requires_review"]
                    ).lower() in {"true", "1", "yes"}
                    qc_clean = not requires_review
                else:
                    qc_clean = True
                finding_group = str(row.get("finding_group", ""))
                rows.append(
                    {
                        "dataset": "NIH-target-domain-private",
                        "patient_id": experiment_id,
                        "study_id": f"{experiment_id}-study",
                        "image_id": experiment_id,
                        "image_path": image_relative.as_posix(),
                        "mask_path": mask_relative.as_posix(),
                        "modality": "CXR",
                        "view": str(row.get("view", "")),
                        "split": split,
                        "sex": str(row.get("sex", "")),
                        "finding_group": finding_group,
                        "label_abnormal": int(finding_group == "abnormal"),
                        "annotation_progress_status": progress_status,
                        "annotation_qc_clean": bool(qc_clean),
                        "image_sha256": sha256(image_target),
                        "mask_sha256": sha256(mask_target),
                    }
                )

        manifest = pd.DataFrame(rows)
        if len(manifest) != 200 or manifest["patient_id"].duplicated().any():
            raise ValueError("The private adaptation manifest is not 200-case disjoint")
        manifest_path = temporary / "private_roi_adaptation_manifest.csv"
        manifest.to_csv(manifest_path, index=False, lineterminator="\n")
        inventory = {
            "artifact": "Private de-identified NIH ROI adaptation bundle",
            "cases": int(len(manifest)),
            "split_counts": {
                str(key): int(value)
                for key, value in manifest["split"].value_counts().sort_index().items()
            },
            "locked_test_qc_clean_cases": int(
                manifest.loc[manifest["split"] == "test", "annotation_qc_clean"].sum()
            ),
            "locked_test_review_flagged_cases": int(
                (~manifest.loc[
                    manifest["split"] == "test", "annotation_qc_clean"
                ]).sum()
            ),
            "patient_level_overlap": 0,
            "preannotations_included": False,
            "prediction_outputs_included": False,
            "risk_metrics_included": False,
            "medical_images_included": True,
            "annotation_masks_included": True,
            "public_upload_allowed": False,
            "manifest_sha256": sha256(manifest_path),
        }
        write_lf(
            temporary / "private_bundle_inventory.json",
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        )
        write_lf(
            temporary / "README_PRIVATE.txt",
            "PRIVATE MEDICAL-IMAGING RESEARCH DATA. DO NOT PUBLISH.\n"
            "Upload only as a private Kaggle dataset and keep sharing disabled.\n",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output).as_posix())
    archive_hash = sha256(archive)
    write_lf(checksum, f"{archive_hash}  {archive.name}\n")
    print("--- PRIVATE ROI ADAPTATION BUNDLE ---")
    print(json.dumps(inventory, indent=2, sort_keys=True))
    print(f"Directory: {output}")
    print(f"ZIP: {archive}")
    print(f"ZIP SHA-256: {archive_hash}")
    print(f"ZIP size MB: {archive.stat().st_size / 1024 / 1024:.2f}")
    print("Original candidate identifiers included: False")
    print("Allowed for public upload: False")


if __name__ == "__main__":
    main()
