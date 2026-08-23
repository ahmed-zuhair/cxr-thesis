from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cxr_thesis.objective1.config import Objective1Config, load_config
from cxr_thesis.objective1.cohort_selection import (
    match_cohort_fingerprints_to_manifest,
    select_blind_projection_recovery_reserves,
    select_projection_replacement_reserves,
    select_roi_annotation_cohort,
)
from cxr_thesis.objective1.annotation_workspace import (
    load_annotation_case,
    load_annotation_worklist,
    load_projection_audit_worklist,
    load_projection_image,
    resolve_annotation_case,
    resolve_projection_audit_cases,
    select_flagged_projection_cases,
    save_binary_annotation,
    update_annotation_progress,
    update_projection_audit,
)
from cxr_thesis.objective1.features import (
    encode_clinical_features,
    extract_handcrafted_2d,
    extract_handcrafted_3d,
)
from cxr_thesis.objective1.graphs import (
    GraphSample,
    build_multimodal_graph,
    build_patch_graph_2d,
    build_patch_graph_3d,
)
from cxr_thesis.objective1.manifest import build_nih_manifest, validate_manifest
from cxr_thesis.objective1.pipeline import run_cxr_manifest
from cxr_thesis.objective1.preprocessing import (
    preprocess_ct_volume,
    resize_with_padding,
    restore_mask,
    transform_mask,
)
from cxr_thesis.objective1.segmentation_data import ROISegmentationDataset
from cxr_thesis.objective1.segmentation import (
    UNet2D,
    dice_score,
    hausdorff95,
    iou_score,
    probability_uncertainty_metrics,
    postprocess_binary_mask,
    remove_small_components,
)
from cxr_thesis.objective1.projection_replacements import (
    finalize_projection_replacements,
    sha256_file,
)


class AnnotationCohortTests(unittest.TestCase):
    @staticmethod
    def _candidate_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        mapping_rows = []
        ranked_rows = []
        number = 0
        for split, count in (("train", 20), ("val", 15)):
            for view in ("PA", "AP"):
                for sex in ("F", "M"):
                    for finding in ("no_finding", "abnormal"):
                        for case in range(count):
                            number += 1
                            image_id = f"image-{number:04d}"
                            mapping_rows.append(
                                {
                                    "candidate_code": f"CASE-{number:04d}",
                                    "patient_id": f"patient-{number:04d}",
                                    "study_id": f"study-{number:04d}",
                                    "image_id": image_id,
                                    "image_path": f"/private/{image_id}.png",
                                    "split": split,
                                    "view_group": view,
                                    "sex_group": sex,
                                    "finding_group": finding,
                                }
                            )
                            risk = number / 1000.0
                            ranked_rows.append(
                                {
                                    "image_id": image_id,
                                    "active_qc_priority_score": risk,
                                    "active_qc_risk_score": risk,
                                    "mask_path": f"/private/{image_id}-mask.png",
                                }
                            )
        return pd.DataFrame(mapping_rows), pd.DataFrame(ranked_rows)

    def test_balanced_disjoint_and_prediction_blind_selection(self) -> None:
        mapping, ranked = self._candidate_frames()
        first = select_roi_annotation_cohort(mapping, ranked, seed=42)

        self.assertEqual(len(first["adaptation_train"]), 120)
        self.assertEqual(len(first["target_validation"]), 40)
        self.assertEqual(len(first["locked_target_test"]), 40)
        self.assertEqual(len(first["master"]), 200)
        self.assertEqual(first["master"]["patient_id"].nunique(), 200)
        self.assertEqual(first["master"]["image_id"].nunique(), 200)
        self.assertTrue((first["adaptation_train"].groupby("cohort_stratum").size() == 15).all())
        self.assertTrue((first["target_validation"].groupby("cohort_stratum").size() == 5).all())
        self.assertTrue((first["locked_target_test"].groupby("cohort_stratum").size() == 5).all())
        self.assertNotIn("mask_path", first["locked_target_test"].columns)
        self.assertNotIn("active_qc_risk_score", first["locked_target_test"].columns)

        reversed_risk = ranked.copy()
        reversed_risk["active_qc_priority_score"] *= -1
        reversed_risk["active_qc_risk_score"] *= -1
        second = select_roi_annotation_cohort(mapping, reversed_risk, seed=42)
        self.assertEqual(
            set(first["locked_target_test"]["image_id"]),
            set(second["locked_target_test"]["image_id"]),
        )

    def test_projection_replacements_preserve_role_stratum_and_basis(self) -> None:
        mapping, ranked = self._candidate_frames()
        roles = select_roi_annotation_cohort(mapping, ranked, seed=42)
        rejected_rows = []
        for role in ("adaptation_train", "target_validation"):
            original = roles[role].iloc[0]
            rejected_rows.append(
                {
                    "candidate_code": original["candidate_code"],
                    "cohort_role": role,
                    "projection_decision": "ineligible_lateral",
                }
            )
        rejected = pd.DataFrame(rejected_rows)
        first = select_projection_replacement_reserves(
            mapping, ranked, roles["master"], rejected, reserves_per_slot=5
        )
        second = select_projection_replacement_reserves(
            mapping, ranked, roles["master"], rejected, reserves_per_slot=5
        )
        self.assertEqual(len(first), 10)
        self.assertEqual(first["replacement_slot"].nunique(), 2)
        self.assertTrue((first.groupby("replacement_slot").size() == 5).all())
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(
            set(first["patient_id"]).intersection(roles["master"]["patient_id"])
        )
        self.assertFalse(
            set(first["image_id"]).intersection(roles["master"]["image_id"])
        )
        for row in rejected_rows:
            original = roles[row["cohort_role"]].loc[
                roles[row["cohort_role"]]["candidate_code"] == row["candidate_code"]
            ].iloc[0]
            reserve = first[first["cohort_role"] == row["cohort_role"]]
            self.assertEqual(set(reserve["cohort_stratum"]), {original["cohort_stratum"]})
            self.assertEqual(set(reserve["selection_basis"]), {original["selection_basis"]})

        forbidden = rejected.copy()
        forbidden.loc[0, "cohort_role"] = "locked_target_test"
        with self.assertRaisesRegex(ValueError, "Only adaptation or validation"):
            select_projection_replacement_reserves(
                mapping, ranked, roles["master"], forbidden
            )

    def test_fingerprint_recovery_and_blind_same_stratum_reserves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_rows = []
            fingerprint_rows = []

            def add_case(
                number: int,
                *,
                split: str,
                view: str,
                sex: str,
                finding: str,
                role: str | None = None,
                decision: str = "eligible_frontal",
            ) -> None:
                path = root / f"image-{number:03d}.png"
                Image.fromarray(
                    np.full((7, 9), number % 255, dtype=np.uint8)
                ).save(path)
                labels = "No Finding" if finding == "no_finding" else "Effusion"
                manifest_rows.append(
                    {
                        "patient_id": f"patient-{number:03d}",
                        "image_id": f"image-{number:03d}",
                        "image_path": str(path),
                        "split": split,
                        "view": view,
                        "sex": sex,
                        "finding_labels": labels,
                    }
                )
                if role is not None:
                    fingerprint_rows.append(
                        {
                            "candidate_code": f"CASE-{number:03d}",
                            "cohort_role": role,
                            "view": view,
                            "sex": sex,
                            "finding_group": finding,
                            "selection_basis": "active_qc_high_risk",
                            "projection_decision": decision,
                            "image_size_bytes": path.stat().st_size,
                            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )

            add_case(
                1,
                split="train",
                view="PA",
                sex="F",
                finding="no_finding",
                role="adaptation_train",
                decision="ineligible_lateral",
            )
            add_case(
                2,
                split="val",
                view="AP",
                sex="M",
                finding="abnormal",
                role="target_validation",
                decision="ineligible_lateral",
            )
            add_case(
                3,
                split="val",
                view="PA",
                sex="M",
                finding="no_finding",
                role="locked_target_test",
            )
            for number in range(10, 16):
                add_case(
                    number,
                    split="train",
                    view="PA",
                    sex="F",
                    finding="no_finding",
                )
            for number in range(20, 26):
                add_case(
                    number,
                    split="val",
                    view="AP",
                    sex="M",
                    finding="abnormal",
                )

            manifest = pd.DataFrame(manifest_rows)
            fingerprints = pd.DataFrame(fingerprint_rows)
            recovered = match_cohort_fingerprints_to_manifest(
                manifest, fingerprints
            )
            self.assertEqual(len(recovered), 3)
            self.assertEqual(recovered["patient_id"].nunique(), 3)
            first = select_blind_projection_recovery_reserves(
                manifest, recovered, seed=42, reserves_per_slot=5
            )
            second = select_blind_projection_recovery_reserves(
                manifest, recovered, seed=42, reserves_per_slot=5
            )
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(len(first), 10)
            self.assertEqual(first["replacement_slot"].nunique(), 2)
            self.assertEqual(first["patient_id"].nunique(), 10)
            self.assertFalse((first["split"] == "test").any())
            self.assertFalse(
                set(first["patient_id"]).intersection(recovered["patient_id"])
            )
            self.assertEqual(
                set(first["replacement_selection_basis"]),
                {"projection_blind_hash"},
            )

            unresolved = recovered.copy()
            unresolved.loc[
                unresolved["cohort_role"] == "locked_target_test",
                "projection_decision",
            ] = "ineligible_lateral"
            with self.assertRaisesRegex(ValueError, "Locked target"):
                select_blind_projection_recovery_reserves(manifest, unresolved)

    def test_projection_replacement_finalization_is_rank_first_and_reversible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort"
            reserves = root / "reserves"
            transaction = root / "transaction"
            identity_rows = []

            def write_original_role(role: str, count: int, rejected: bool) -> None:
                role_root = cohort / role
                for folder in ("images", "preannotations", "annotations"):
                    (role_root / folder).mkdir(parents=True, exist_ok=True)
                rows = []
                audits = []
                for index in range(1, count + 1):
                    code = f"{role}-original-{index}"
                    Image.fromarray(np.full((8, 10), index, dtype=np.uint8)).save(
                        role_root / "images" / f"{code}.png"
                    )
                    if role != "locked_target_test":
                        Image.fromarray(
                            np.full((8, 10), 255, dtype=np.uint8)
                        ).save(role_root / "preannotations" / f"{code}.png")
                    rows.append(
                        {
                            "candidate_code": code,
                            "cohort_role": role,
                            "view": "PA",
                            "sex": "F",
                            "finding_group": "abnormal",
                            "selection_basis": "active_qc_high_risk",
                            "image_filename": f"images/{code}.png",
                            "preannotation_filename": (
                                f"preannotations/{code}.png"
                                if role != "locked_target_test"
                                else ""
                            ),
                            "required_output_mask": f"annotations/{code}.png",
                        }
                    )
                    decision = (
                        "ineligible_lateral" if rejected and index == 1 else "eligible_frontal"
                    )
                    audits.append(
                        {
                            "candidate_code": code,
                            "cohort_role": role,
                            "projection_decision": decision,
                            "auditor": "tester",
                            "updated_utc": "2026-01-01T00:00:00+00:00",
                            "note": "",
                        }
                    )
                pd.DataFrame(rows).to_csv(
                    role_root / "annotation_worklist.csv", index=False
                )
                pd.DataFrame(audits).to_csv(
                    role_root / "projection_audit.csv", index=False
                )

            write_original_role("adaptation_train", 3, True)
            write_original_role("target_validation", 2, True)
            write_original_role("locked_target_test", 2, False)
            locked_hash_before = sha256_file(
                cohort / "locked_target_test" / "annotation_worklist.csv"
            )

            for slot, role in enumerate(
                ("adaptation_train", "target_validation"), start=1
            ):
                role_root = reserves / role
                for folder in ("images", "preannotations", "annotations"):
                    (role_root / folder).mkdir(parents=True, exist_ok=True)
                rows = []
                audits = []
                original_code = f"{role}-original-1"
                for rank in range(1, 6):
                    code = f"{role}-reserve-{rank}"
                    Image.fromarray(
                        np.full((8, 10), 20 + rank, dtype=np.uint8)
                    ).save(role_root / "images" / f"{code}.png")
                    Image.fromarray(
                        np.full((8, 10), 255, dtype=np.uint8)
                    ).save(role_root / "preannotations" / f"{code}.png")
                    rows.append(
                        {
                            "candidate_code": code,
                            "cohort_role": role,
                            "view": "PA",
                            "sex": "F",
                            "finding_group": "abnormal",
                            "selection_basis": "active_qc_high_risk",
                            "replacement_selection_basis": "projection_blind_hash",
                            "replacement_slot": f"RPL-{slot:02d}",
                            "reserve_rank": rank,
                            "image_filename": f"images/{code}.png",
                            "preannotation_filename": f"preannotations/{code}.png",
                            "required_output_mask": f"annotations/{code}.png",
                        }
                    )
                    audits.append(
                        {
                            "candidate_code": code,
                            "cohort_role": role,
                            "projection_decision": "eligible_frontal",
                            "auditor": "tester",
                            "updated_utc": "2026-01-02T00:00:00+00:00",
                            "note": "",
                        }
                    )
                    identity_rows.append(
                        {
                            "replacement_code": code,
                            "candidate_code": original_code,
                            "cohort_role": role,
                            "patient_id": f"patient-{slot}-{rank}",
                            "image_id": f"image-{slot}-{rank}",
                            "replacement_slot": f"RPL-{slot:02d}",
                            "reserve_rank": rank,
                        }
                    )
                pd.DataFrame(rows).to_csv(
                    role_root / "annotation_worklist.csv", index=False
                )
                pd.DataFrame(audits).to_csv(
                    role_root / "projection_audit.csv", index=False
                )

            pd.DataFrame(identity_rows).to_csv(
                reserves / "replacement_identity_private.csv", index=False
            )
            (reserves / "replacement_recovery_summary_private.json").write_text(
                json.dumps(
                    {
                        "official_nih_test_used": False,
                        "locked_target_test_modified": False,
                        "replacement_selection_uses_predictions": False,
                    }
                ),
                encoding="utf-8",
            )

            summary = finalize_projection_replacements(
                cohort, reserves, transaction
            )
            self.assertEqual(summary["selected_replacements"], 2)
            self.assertFalse(summary["locked_target_test_modified"])
            self.assertEqual(
                locked_hash_before,
                sha256_file(cohort / "locked_target_test" / "annotation_worklist.csv"),
            )
            for role, expected_count in (
                ("adaptation_train", 3),
                ("target_validation", 2),
            ):
                role_root = cohort / role
                worklist = pd.read_csv(role_root / "annotation_worklist.csv")
                audit = pd.read_csv(role_root / "projection_audit.csv")
                self.assertEqual(len(worklist), expected_count)
                self.assertNotIn(f"{role}-original-1", set(worklist["candidate_code"]))
                self.assertIn(f"{role}-reserve-1", set(worklist["candidate_code"]))
                self.assertNotIn(f"{role}-reserve-2", set(worklist["candidate_code"]))
                self.assertEqual(set(audit["projection_decision"]), {"eligible_frontal"})
                self.assertTrue(
                    (role_root / "images" / f"{role}-reserve-1.png").is_file()
                )
                self.assertTrue(
                    (role_root / "preannotations" / f"{role}-reserve-1.png").is_file()
                )
            self.assertTrue(
                (transaction / "applied_projection_replacements_private.csv").is_file()
            )
            self.assertTrue(
                (
                    transaction
                    / "before_replacement"
                    / "adaptation_train"
                    / "annotation_worklist.csv"
                ).is_file()
            )


class AnnotationWorkspaceTests(unittest.TestCase):
    @staticmethod
    def _write_role(root: Path, role: str, *, with_preannotation: bool) -> Path:
        role_root = root / role
        (role_root / "images").mkdir(parents=True)
        (role_root / "annotations").mkdir()
        image = np.arange(48, dtype=np.uint8).reshape(6, 8)
        Image.fromarray(image).save(role_root / "images" / "CASE-1.png")
        preannotation = ""
        if with_preannotation:
            (role_root / "preannotations").mkdir()
            Image.fromarray((image > 20).astype(np.uint8) * 255).save(
                role_root / "preannotations" / "CASE-1.png"
            )
            preannotation = "preannotations/CASE-1.png"
        pd.DataFrame(
            [
                {
                    "candidate_code": "CASE-1",
                    "cohort_role": role,
                    "image_filename": "images/CASE-1.png",
                    "preannotation_filename": preannotation,
                    "required_output_mask": "annotations/CASE-1.png",
                }
            ]
        ).to_csv(role_root / "annotation_worklist.csv", index=False)
        return role_root

    def test_preannotation_save_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            role_root = self._write_role(root, "adaptation_train", with_preannotation=True)
            worklist, resolved_root = load_annotation_worklist(root, "adaptation_train")
            case = resolve_annotation_case(
                worklist.iloc[0], resolved_root, role="adaptation_train"
            )
            image, mask, source = load_annotation_case(case)
            self.assertEqual(source, "preannotation")
            self.assertEqual(image.shape, mask.shape)
            metrics = save_binary_annotation(mask, case.output_path, expected_shape=image.shape)
            self.assertTrue(case.output_path.is_file())
            self.assertGreater(metrics["foreground_fraction"], 0.0)
            progress = update_annotation_progress(
                role_root / "annotation_progress.csv",
                candidate_code="CASE-1",
                role="adaptation_train",
                annotator="tester",
                foreground_fraction=float(metrics["foreground_fraction"]),
                needs_review=False,
                note="checked",
            )
            self.assertEqual(progress.iloc[0]["status"], "complete")

    def test_locked_test_rejects_preannotations_and_loads_blank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_role(root, "locked_target_test", with_preannotation=False)
            worklist, role_root = load_annotation_worklist(root, "locked_target_test")
            case = resolve_annotation_case(
                worklist.iloc[0], role_root, role="locked_target_test"
            )
            image, mask, source = load_annotation_case(case)
            self.assertEqual(source, "blank_prediction_blind")
            self.assertEqual(int(mask.sum()), 0)
            self.assertEqual(image.shape, mask.shape)

            (role_root / "preannotations").mkdir()
            Image.fromarray(np.zeros(image.shape, dtype=np.uint8)).save(
                role_root / "preannotations" / "forbidden.png"
            )
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                load_annotation_worklist(root, "locked_target_test")

    def test_projection_audit_is_image_only_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            role_root = self._write_role(root, "adaptation_train", with_preannotation=True)
            worklist = pd.read_csv(role_root / "annotation_worklist.csv")
            worklist["preannotation_filename"] = "missing-prediction-must-not-be-read.png"
            worklist.to_csv(role_root / "annotation_worklist.csv", index=False)

            frame, resolved_root = load_projection_audit_worklist(
                root, "adaptation_train"
            )
            cases = resolve_projection_audit_cases(
                frame, resolved_root, role="adaptation_train"
            )
            image = load_projection_image(cases[0])
            self.assertEqual(image.shape, (6, 8))
            audit = update_projection_audit(
                role_root / "projection_audit.csv",
                candidate_code="CASE-1",
                role="adaptation_train",
                auditor="tester",
                decision="ineligible_lateral",
                note="visual lateral",
            )
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit.iloc[0]["projection_decision"], "ineligible_lateral")
            flagged = select_flagged_projection_cases(
                cases, role_root / "projection_audit.csv"
            )
            self.assertEqual([case.candidate_code for case in flagged], ["CASE-1"])
            update_projection_audit(
                role_root / "projection_audit.csv",
                candidate_code="CASE-1",
                role="adaptation_train",
                auditor="second-reviewer",
                decision="eligible_frontal",
                note="confirmed frontal",
            )
            self.assertEqual(
                select_flagged_projection_cases(
                    cases, role_root / "projection_audit.csv"
                ),
                [],
            )
            with self.assertRaisesRegex(ValueError, "Unsupported projection decision"):
                update_projection_audit(
                    role_root / "projection_audit.csv",
                    candidate_code="CASE-1",
                    role="adaptation_train",
                    auditor="tester",
                    decision="guess",
                    note="",
                )


class ManifestTests(unittest.TestCase):
    def test_patient_leakage_is_rejected(self) -> None:
        frame = pd.DataFrame(
            [
                dict(dataset="demo", patient_id="p1", study_id="s1", image_id="i1", image_path="a.png", modality="CXR", view="PA", split="train"),
                dict(dataset="demo", patient_id="p1", study_id="s2", image_id="i2", image_path="b.png", modality="CXR", view="PA", split="test"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "Patient leakage"):
            validate_manifest(frame)

    def test_build_nih_manifest_preserves_patient_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            train_names, test_names = [], []
            for patient in range(1, 6):
                name = f"{patient:08d}_000.png"
                rows.append(
                    {
                        "Image Index": name,
                        "Finding Labels": "Effusion" if patient % 2 else "No Finding",
                        "Follow-up #": 0,
                        "Patient ID": patient,
                        "Patient Age": 50 + patient,
                        "Patient Gender": "M" if patient % 2 else "F",
                        "View Position": "PA",
                        "OriginalImagePixelSpacing_x": 0.14,
                        "OriginalImagePixelSpacing_y": 0.14,
                    }
                )
                (test_names if patient == 5 else train_names).append(name)
            pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
            (root / "train.txt").write_text("\n".join(train_names), encoding="utf-8")
            (root / "test.txt").write_text("\n".join(test_names), encoding="utf-8")
            frame = build_nih_manifest(
                root / "metadata.csv",
                root / "train.txt",
                root / "test.txt",
                root / "images",
                val_fraction=0.25,
                seed=7,
            )
            summary = validate_manifest(frame)
            self.assertEqual(summary["rows"], 5)
            self.assertEqual(frame.loc[frame.image_id.str.contains("00000005"), "split"].item(), "test")
            self.assertEqual(set(frame.split), {"train", "val", "test"})
            self.assertIn("pixel_spacing_x", frame.columns)

    def test_build_nih_manifest_resolves_kaggle_nested_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "images_001" / "images"
            nested.mkdir(parents=True)
            rows = []
            for patient in range(1, 4):
                name = f"{patient:08d}_000.png"
                Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(nested / name)
                rows.append(
                    {
                        "Image Index": name,
                        "Finding Labels": "No Finding",
                        "Follow-up #": 0,
                        "Patient ID": patient,
                    }
                )
            pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
            (root / "train.txt").write_text("00000001_000.png\n00000002_000.png\n", encoding="utf-8")
            (root / "test.txt").write_text("00000003_000.png\n", encoding="utf-8")
            frame = build_nih_manifest(
                root / "metadata.csv",
                root / "train.txt",
                root / "test.txt",
                root,
                val_fraction=0.5,
            )
            self.assertTrue(all(Path(path).is_file() for path in frame["image_path"]))


class PreprocessingTests(unittest.TestCase):
    def test_resize_preserves_aspect_and_mask_alignment(self) -> None:
        image = np.arange(50 * 100, dtype=np.uint16).reshape(50, 100)
        resized, geometry = resize_with_padding(image, (200, 200))
        mask = np.ones_like(image, dtype=np.uint8)
        transformed = transform_mask(mask, geometry)
        restored = restore_mask(transformed, geometry)
        self.assertEqual(resized.shape, (200, 200))
        self.assertEqual((geometry.resized_height, geometry.resized_width), (100, 200))
        self.assertEqual(int(transformed.sum()), 100 * 200)
        self.assertFalse(transformed[:50].any())
        np.testing.assert_array_equal(restored, mask)

    def test_ct_window_and_resampling(self) -> None:
        volume = np.linspace(-1200, 600, 4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
        output, metadata = preprocess_ct_volume(volume, (2.0, 1.0, 1.0), Objective1Config().preprocessing)
        self.assertEqual(output.shape, (8, 8, 8))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)
        self.assertEqual(metadata["target_spacing"], [1.0, 1.0, 1.0])


class SegmentationTests(unittest.TestCase):
    def test_metrics(self) -> None:
        target = np.zeros((32, 32), dtype=np.uint8)
        target[8:24, 8:24] = 1
        self.assertAlmostEqual(dice_score(target, target), 1.0)
        self.assertAlmostEqual(iou_score(target, target), 1.0)
        self.assertAlmostEqual(hausdorff95(target, target), 0.0)

    def test_unet_output_and_postprocessing(self) -> None:
        model = UNet2D(channels=(4, 8, 16, 32))
        output = model(torch.randn(2, 1, 65, 71))
        self.assertEqual(tuple(output.shape), (2, 1, 65, 71))
        probability = np.zeros((32, 32), dtype=np.float32)
        probability[5:20, 5:20] = 0.9
        probability[28:30, 28:30] = 0.9
        mask = postprocess_binary_mask(probability, Objective1Config().segmentation)
        self.assertTrue(mask[10, 10])

    def test_relative_component_cleanup_preserves_meaningful_third_region(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[2:22, 2:22] = 1
        mask[2:20, 30:50] = 1
        mask[30:35, 2:6] = 1
        mask[60, 60] = 1
        cleaned, audit = remove_small_components(
            mask,
            min_component_fraction=0.02,
        )
        self.assertTrue(cleaned[10, 10])
        self.assertTrue(cleaned[10, 40])
        self.assertTrue(cleaned[32, 3])
        self.assertFalse(cleaned[60, 60])
        self.assertEqual(audit["components_before"], 4)
        self.assertEqual(audit["components_after"], 3)
        self.assertEqual(audit["removed_pixels"], 1)

    def test_probability_uncertainty_detects_ambiguous_predictions(self) -> None:
        confident = np.zeros((32, 32), dtype=np.float32)
        confident[:, :16] = 0.99
        confident[:, 16:] = 0.01
        ambiguous = np.full((32, 32), 0.54, dtype=np.float32)
        ambiguous[:, :16] = 0.56
        confident_metrics = probability_uncertainty_metrics(
            confident,
            threshold=0.55,
            margin=0.10,
        )
        ambiguous_metrics = probability_uncertainty_metrics(
            ambiguous,
            threshold=0.55,
            margin=0.10,
        )
        self.assertLess(
            confident_metrics["mean_binary_entropy"],
            ambiguous_metrics["mean_binary_entropy"],
        )
        self.assertEqual(confident_metrics["uncertain_fraction"], 0.0)
        self.assertEqual(ambiguous_metrics["uncertain_fraction"], 1.0)
        self.assertGreater(
            ambiguous_metrics["boundary_entropy_mean"],
            confident_metrics["boundary_entropy_mean"],
        )

    def test_batched_mask_generation_writes_auditable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(2):
                image_path = root / f"image-{index}.png"
                image = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
                Image.fromarray(image).save(image_path)
                rows.append(
                    dict(
                        dataset="demo",
                        patient_id=f"p{index}",
                        study_id=f"s{index}",
                        image_id=f"i{index}",
                        image_path=str(image_path),
                        mask_path="",
                        modality="CXR",
                        view="PA",
                        split="external",
                    )
                )
            manifest_path = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            checkpoint_path = root / "checkpoint.pt"
            model = UNet2D(channels=(4, 8, 16, 32))
            torch.save(
                {
                    "architecture": "UNet2D",
                    "channels": [4, 8, 16, 32],
                    "epoch": 1,
                    "model_state": model.state_dict(),
                    "validation_metrics": {"threshold": 0.55},
                },
                checkpoint_path,
            )
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            repository = Path(__file__).resolve().parents[1]
            output_manifest = root / "output.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(repository / "scripts" / "generate_roi_masks.py"),
                    "--manifest",
                    str(manifest_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--config",
                    str(repository / "configs" / "objective1" / "default.yaml"),
                    "--mask-dir",
                    str(root / "masks"),
                    "--output-manifest",
                    str(output_manifest),
                    "--batch-size",
                    "2",
                    "--device",
                    "cpu",
                    "--expected-checkpoint-sha256",
                    digest,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            output = pd.read_csv(output_manifest)
            audit = pd.read_csv(root / "output_audit.csv")
            summary = pd.read_json(root / "output_summary.json", typ="series")
            self.assertEqual(int((output["mask_generation_status"] == "complete").sum()), 2)
            self.assertEqual(len(audit), 2)
            self.assertEqual(int(summary["generated_this_run"]), 2)
            self.assertEqual(summary["checkpoint_sha256"], digest)
            self.assertIn("mean_binary_entropy", audit.columns)
            self.assertIn("uncertain_fraction", audit.columns)
            self.assertIn("mean_boundary_entropy", summary.index)
            resumed = subprocess.run(
                [
                    *result.args,
                    "--resume",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, msg=resumed.stderr)
            resumed_summary = pd.read_json(root / "output_summary.json", typ="series")
            self.assertEqual(int(resumed_summary["generated_this_run"]), 0)
            self.assertEqual(int(resumed_summary["resumed_this_run"]), 2)


class FeatureAndGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        y, x = np.ogrid[:64, :64]
        self.image = (x + y).astype(np.float32)
        self.mask = ((x - 32) ** 2 / 24**2 + (y - 32) ** 2 / 28**2 <= 1).astype(np.uint8)

    def test_feature_families(self) -> None:
        features = extract_handcrafted_2d(self.image, self.mask)
        self.assertIn("roi_lbp_hist_00", features)
        self.assertIn("roi_hog_hist_00", features)
        self.assertIn("roi_left_right_asymmetry", features)
        clinical = encode_clinical_features({"age": 60, "sex": "F", "view": "PA"})
        self.assertEqual(clinical["clinical_sex_female"], 1.0)
        volume = np.stack([self.image] * 4)
        mask3d = np.stack([self.mask] * 4)
        features3d = extract_handcrafted_3d(volume, mask3d, (2.0, 1.0, 1.0))
        self.assertGreater(features3d["roi3d_volume_mm3"], 0)

    def test_2d_3d_and_multimodal_graphs(self) -> None:
        graph2d = build_patch_graph_2d(self.image, self.mask, grid=(4, 4), knn_k=2)
        graph2d.validate()
        self.assertEqual(graph2d.node_position.shape[1], 2)
        volume = np.stack([self.image] * 8)
        mask3d = np.stack([self.mask] * 8)
        graph3d = build_patch_graph_3d(volume, mask3d, grid=(2, 4, 4), knn_k=1)
        graph3d.validate()
        self.assertEqual(graph3d.node_position.shape[1], 3)
        multimodal = build_multimodal_graph(
            {"left_lung": np.ones(8), "right_lung": np.zeros(8)},
            {"texture": 0.2, "shape": 0.4},
            {"age": 0.5, "view": 1.0},
        )
        self.assertEqual(set(multimodal.node_type), {"roi", "radiomics", "clinical"})
        with tempfile.TemporaryDirectory() as directory:
            path = multimodal.save(Path(directory) / "graph.npz")
            loaded = GraphSample.load(path)
            np.testing.assert_array_equal(multimodal.edge_index, loaded.edge_index)

    def test_end_to_end_manifest_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(self.image.astype(np.uint8)).save(image_path)
            Image.fromarray(self.mask.astype(np.uint8) * 255).save(mask_path)
            frame = pd.DataFrame(
                [
                    dict(
                        dataset="demo",
                        patient_id="p1",
                        study_id="s1",
                        image_id="i1",
                        image_path=str(image_path),
                        mask_path=str(mask_path),
                        modality="CXR",
                        view="PA",
                        split="train",
                        age=45,
                        sex="F",
                    )
                ]
            )
            output = run_cxr_manifest(frame, Objective1Config(), root / "derived")
            self.assertEqual(len(output), 1)
            self.assertTrue((root / "derived" / "graphs" / "i1.npz").is_file())
            self.assertTrue((root / "derived" / "features.csv").is_file())
            dataset = ROISegmentationDataset(
                frame,
                root,
                Objective1Config().preprocessing,
                split="train",
            )
            image_tensor, mask_tensor, image_id = dataset[0]
            self.assertEqual(tuple(image_tensor.shape), (1, 224, 224))
            self.assertEqual(tuple(mask_tensor.shape), (1, 224, 224))
            self.assertEqual(image_id, "i1")


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "objective1" / "default.yaml")
        self.assertEqual(config.graph.patch_grid_2d, (7, 7))


if __name__ == "__main__":
    unittest.main()
