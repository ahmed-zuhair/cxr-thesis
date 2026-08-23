"""Create a private image-only review bundle of deterministic replacement reserves."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.cohort_selection import (
    select_projection_replacement_reserves,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build private same-stratum projection-replacement reserves"
    )
    parser.add_argument("--private-mapping", required=True)
    parser.add_argument("--ranked-audit", required=True)
    parser.add_argument("--current-cohort", required=True)
    parser.add_argument("--replacement-request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--reserves-per-slot", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping_path = Path(args.private_mapping).resolve()
    ranked_path = Path(args.ranked_audit).resolve()
    cohort_path = Path(args.current_cohort).resolve()
    request_path = Path(args.replacement_request).resolve()
    output = Path(args.output_dir).resolve()
    bundle = Path(args.bundle).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    mapping = pd.read_csv(mapping_path)
    ranked = pd.read_csv(ranked_path)
    cohort = pd.read_csv(cohort_path)
    request = pd.read_csv(request_path)
    reserves = select_projection_replacement_reserves(
        mapping,
        ranked,
        cohort,
        request,
        reserves_per_slot=args.reserves_per_slot,
    )

    private_manifest = output / "replacement_reserves_private.csv"
    reserves.to_csv(private_manifest, index=False)
    worklist_parts: list[pd.DataFrame] = []
    for role, frame in reserves.groupby("cohort_role", sort=True):
        role_root = output / str(role)
        image_dir = role_root / "images"
        preannotation_dir = role_root / "preannotations"
        annotation_dir = role_root / "annotations"
        image_dir.mkdir(parents=True)
        preannotation_dir.mkdir()
        annotation_dir.mkdir()
        rows: list[dict[str, object]] = []
        for _, row in frame.sort_values(
            ["replacement_slot", "reserve_rank"], kind="stable"
        ).iterrows():
            code = str(row["candidate_code"])
            source_image = Path(str(row["image_path"]))
            source_mask = Path(str(row["mask_path"]))
            if not source_image.is_file():
                raise FileNotFoundError(f"Replacement source image missing: {source_image}")
            if not source_mask.is_file():
                raise FileNotFoundError(f"Replacement mask missing: {source_mask}")
            image_target = image_dir / f"{code}.png"
            mask_target = preannotation_dir / f"{code}.png"
            shutil.copy2(source_image, image_target)
            shutil.copy2(source_mask, mask_target)
            with Image.open(image_target) as image_handle, Image.open(mask_target) as mask_handle:
                if image_handle.size != mask_handle.size:
                    raise RuntimeError(f"Image/mask dimensions differ for reserve {code}")
            rows.append(
                {
                    "candidate_code": code,
                    "cohort_role": str(role),
                    "replacement_slot": str(row["replacement_slot"]),
                    "reserve_rank": int(row["reserve_rank"]),
                    "image_filename": f"images/{code}.png",
                    "preannotation_filename": f"preannotations/{code}.png",
                    "required_output_mask": f"annotations/{code}.png",
                }
            )
        worklist = pd.DataFrame(rows)
        worklist.to_csv(role_root / "annotation_worklist.csv", index=False)
        worklist_parts.append(worklist)

    combined_worklist = pd.concat(worklist_parts, ignore_index=True)
    summary = {
        "purpose": "private prediction-blind projection review of replacement reserves",
        "replacement_slots": int(reserves["replacement_slot"].nunique()),
        "reserves_per_slot": int(args.reserves_per_slot),
        "total_reserves": int(len(reserves)),
        "role_counts": {
            str(key): int(value)
            for key, value in reserves["cohort_role"].value_counts().items()
        },
        "locked_target_test_modified": False,
        "official_nih_test_used": False,
        "patient_overlap_with_current_cohort": int(
            len(set(reserves["patient_id"]).intersection(cohort["patient_id"]))
        ),
        "image_overlap_with_current_cohort": int(
            len(set(reserves["image_id"]).intersection(cohort["image_id"]))
        ),
        "private_mapping_sha256": sha256_file(mapping_path),
        "ranked_audit_sha256": sha256_file(ranked_path),
        "current_cohort_sha256": sha256_file(cohort_path),
        "replacement_request_sha256": sha256_file(request_path),
        "private_manifest_sha256": sha256_file(private_manifest),
        "identifiers_allowed_for_publication": False,
        "images_or_masks_allowed_for_publication": False,
    }
    (output / "replacement_reserve_summary_private.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if bundle.exists():
        raise FileExistsError(f"Bundle already exists: {bundle}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output).as_posix())
    bundle_hash = sha256_file(bundle)
    bundle.with_suffix(".sha256").write_text(
        f"{bundle_hash}  {bundle.name}\n", encoding="utf-8"
    )

    print("--- PRIVATE REPLACEMENT RESERVE BUNDLE ---")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Worklist rows: {len(combined_worklist)}")
    print(f"Bundle: {bundle}")
    print(f"Bundle size MB: {bundle.stat().st_size / (1024 * 1024):.2f}")
    print(f"Bundle SHA-256: {bundle_hash}")
    print("Candidate identifiers displayed: False")
    print("Allowed for public upload: False")
    print("PRIVATE PROJECTION REPLACEMENT RESERVES READY")


if __name__ == "__main__":
    main()
