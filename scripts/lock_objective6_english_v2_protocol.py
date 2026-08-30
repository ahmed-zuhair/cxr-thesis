#!/usr/bin/env python3
"""Lock the separate Objective 6 English fact-aware v2 extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from cxr_thesis.objective6.v2 import (
    canonical_private_patient,
    select_development_patients,
)

TRAIN_SHA256 = "278addf3c0a216bb206b4e4b79364f26bacbee977f3209e9275e2abbd8fda7d7"
PRETRAINING_LOCK_SHA256 = "9c66e494f3247aa00782133d671e004ded0a28d4a9626491279ff65d36b0aa6d"
SOURCE_CLASSIFIER_SHA256 = "109db89a723c6e2f24442cb5866bfcf4084e85083936cda91bce3b8ae4365d9d"
VERSION = "v2.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--pretraining-lock", type=Path, required=True)
    parser.add_argument("--v1-1-summary", type=Path, required=True)
    parser.add_argument("--v1-1-final-lock", type=Path, required=True)
    parser.add_argument("--expected-v1-1-summary-sha256", required=True)
    parser.add_argument("--expected-v1-1-lock-sha256", required=True)
    parser.add_argument("--translator-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--seed", type=int, default=6242)
    parser.add_argument("--development-fraction", type=float, default=0.20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: dict[str, Any], path: Path) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def validate_v1_1(summary: dict[str, Any], final_lock: dict[str, Any]) -> None:
    if summary.get("artifact") != "Objective 6 locked v1.1 enhancement validation comparison":
        raise RuntimeError("Unexpected Objective 6 v1.1 summary")
    if (
        summary.get("all_conditions_passed") is not False
        or summary.get("advance_to_single_locked_test_evaluation") is not False
        or summary.get("additional_enhancement_rounds_allowed") is not False
        or summary.get("locked_test_manifest_opened") is not False
        or summary.get("locked_test_reports_accessed") is not False
        or summary.get("locked_test_evaluated") is not False
    ):
        raise RuntimeError("Objective 6 v1.1 is not in the locked negative state")
    if (
        final_lock.get("artifact") != "Final Objective 6 v1.1 validation decision lock"
        or final_lock.get("immutable") is not True
        or final_lock.get("all_advancement_conditions_passed") is not False
        or final_lock.get("advance_to_single_locked_test_evaluation") is not False
        or final_lock.get("additional_enhancement_rounds_allowed") is not False
        or final_lock.get("locked_test_evaluated") is not False
        or final_lock.get("locked_test_evaluation_count") != 0
    ):
        raise RuntimeError("Objective 6 v1.1 final lock changed")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists and will not be overwritten: {args.output_dir}")
    protected = {
        args.train_manifest: TRAIN_SHA256,
        args.pretraining_lock: PRETRAINING_LOCK_SHA256,
        args.v1_1_summary: args.expected_v1_1_summary_sha256,
        args.v1_1_final_lock: args.expected_v1_1_lock_sha256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 input changed: {path}")
    revision = args.translator_revision.casefold()
    if len(revision) < 7 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Translator revision must be an immutable hexadecimal commit")

    summary = json.loads(args.v1_1_summary.read_text(encoding="utf-8"))
    final_lock = json.loads(args.v1_1_final_lock.read_text(encoding="utf-8"))
    validate_v1_1(summary, final_lock)

    source = pd.read_csv(args.train_manifest, low_memory=False)
    required = {"patient_id", "report", "image_path", "labels", "split"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Objective 6 training manifest columns are missing: {missing}")
    if len(source) != 29283 or set(source["split"].astype(str)) != {"train"}:
        raise RuntimeError("Objective 6 source training cohort changed")
    source["patient_id"] = source["patient_id"].map(canonical_private_patient)
    development_patients = select_development_patients(
        source["patient_id"], seed=args.seed, fraction=args.development_fraction
    )
    development = source[source["patient_id"].isin(development_patients)].copy()
    training = source[~source["patient_id"].isin(development_patients)].copy()
    if not len(training) or not len(development):
        raise RuntimeError("Objective 6 v2 partition produced an empty role")
    overlap = set(training["patient_id"]) & set(development["patient_id"])
    if overlap or len(training) + len(development) != len(source):
        raise RuntimeError("Objective 6 v2 patient partition is invalid")
    training["split"] = "v2_train"
    development["split"] = "v2_development"

    public = args.output_dir / "public"
    private = args.output_dir / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    private_paths = {
        "v2_train": private / "v2_train_report_cohort_private.csv",
        "v2_development": private / "v2_development_report_cohort_private.csv",
    }
    training.to_csv(private_paths["v2_train"], index=False, lineterminator="\n")
    development.to_csv(private_paths["v2_development"], index=False, lineterminator="\n")
    private_hashes = {role: sha256(path) for role, path in private_paths.items()}
    for role, path in private_paths.items():
        path.with_suffix(".csv.sha256").write_text(
            f"{private_hashes[role]}  {path.name}\n", encoding="utf-8"
        )

    protocol: dict[str, Any] = {
        "artifact": "Objective 6 English fact-aware report-generation v2 protocol",
        "version": VERSION,
        "status": (
            "locked after the immutable negative v1.1 decision and before v2 "
            "translation, model fitting, generation, or locked-test access"
        ),
        "repository_commit_before_v2_implementation": args.repository_commit,
        "scientific_separation": {
            "v1_and_v1_1_results_retained": True,
            "v1_1_negative_result_rewritten": False,
            "v1_1_additional_round_prohibition_respected": True,
            "v2_is_a_separate_exploratory_remedial_extension": True,
            "v2_uses_the_previous_6280_case_validation_for_tuning": False,
            "v2_uses_the_original_locked_test_for_tuning": False,
            "rationale": (
                "v2 changes the report language, cohort-development boundary, "
                "clinical planner, retrieval mechanism, decoder pretraining, and "
                "factual objective; it is not represented as a continuation of v1.1"
            ),
        },
        "evidence_basis": [
            {"method": "fact-aware multimodal retrieval augmentation", "source": "https://aclanthology.org/2025.naacl-long.28/"},
            {"method": "classifier-plan and template replacement", "source": "https://aclanthology.org/2023.findings-acl.683/"},
            {"method": "reinforcement learning with factual rewards", "source": "https://aclanthology.org/2021.naacl-main.416/"},
            {"method": "bilingual grounded PadChest supervision", "source": "https://arxiv.org/abs/2411.05085"},
        ],
        "private_v2_development_partition": {
            "source": "the original Objective 6 training cohort only",
            "source_cases": len(source),
            "source_patients": int(source["patient_id"].nunique()),
            "algorithm": (
                "rank unique private patient identifiers by SHA-256 of "
                "objective6-v2|seed|patient and assign the first rounded 20 percent "
                "to development"
            ),
            "seed": args.seed,
            "requested_development_fraction": args.development_fraction,
            "training_cases": len(training),
            "training_patients": int(training["patient_id"].nunique()),
            "development_cases": len(development),
            "development_patients": int(development["patient_id"].nunique()),
            "patient_overlap": 0,
            "selection_used_labels": False,
            "selection_used_report_content": False,
            "selection_used_images": False,
            "selection_used_predictions": False,
            "private_columns_copied_without_content_statistics": True,
            "private_manifest_sha256": private_hashes,
        },
        "english_reference_pipeline": {
            "source_language": "Spanish",
            "target_language": "English",
            "execution": "local private Kaggle inference; no translation API",
            "model": "facebook/nllb-200-distilled-600M",
            "immutable_model_revision": revision,
            "source_token": "spa_Latn",
            "target_token": "eng_Latn",
            "decoding": {"method": "deterministic beam search", "beam_width": 5, "sampling": False, "maximum_new_tokens": 256},
            "normalisation": (
                "Unicode NFC, whitespace normalization, preserve measurements and "
                "remove neither negations nor uncertainty expressions"
            ),
            "translation_quality_gate": {
                "development_only": True,
                "minimum_PadChest6_concept_polarity_agreement": 0.95,
                "minimum_nonempty_fraction": 0.995,
                "maximum_number_or_measurement_loss_rate": 0.01,
                "manual_or_locked_test_feedback_used": False,
            },
            "PadChest_GR_if_licensed_and_available": (
                "may be used only as external bilingual training supervision and "
                "translation validation after patient de-duplication; absence does "
                "not change the locked primary pipeline"
            ),
            "Spanish_reports_public": False,
            "English_reports_public": False,
        },
        "v2_candidates": {
            "candidate_count": 2,
            "candidate_1": {
                "name": "fact_aware_retrieve_edit_english",
                "query_information": ["frozen image embedding", "predicted clinical probabilities", "age", "sex", "projection"],
                "retrieval_bank": "v2 training patients only",
                "initial_image_neighbors": 32,
                "final_fact_reranked_neighbors": 8,
                "reranking_score": {"image_cosine_similarity": 0.45, "predicted_concept_agreement": 0.45, "non_diagnostic_metadata_similarity": 0.10},
                "report_controller": (
                    "sentence-level English retrieve-and-edit with predicted-positive "
                    "insertion, predicted-negative contradiction removal, and an "
                    "uncertainty-preserving abstention rule"
                ),
            },
            "candidate_2": {
                "name": "fact_aware_retrieval_augmented_english_generator",
                "planner": (
                    "multi-label clinical plan trained on v2-training PadChest/UMLS "
                    "labels; ground-truth labels are never decoder inputs at inference"
                ),
                "visual_encoder": (
                    "Objective 5 PadChest-adapted DenseNet-121 initialized from "
                    f"SHA-256 {SOURCE_CLASSIFIER_SHA256}"
                ),
                "text_backbone": "English-capable pretrained encoder-decoder with LoRA",
                "decoder_inputs": ["visual tokens", "clinical-plan graph token", "non-diagnostic clinical token", "retrieved English exemplars"],
                "objective": {
                    "report_cross_entropy": 1.0,
                    "clinical_plan_binary_cross_entropy": 0.40,
                    "image_report_contrastive_alignment": 0.20,
                    "factual_consistency_loss": 0.20,
                    "contradiction_and_repetition_unlikelihood": 0.05,
                },
                "final_stage": (
                    "one short self-critical sequence-training stage using a locked "
                    "clinical factuality, semantic similarity, CIDEr-D, contradiction, "
                    "and diversity reward"
                ),
            },
        },
        "development_evaluation": {
            "evaluation_repetitions": 1,
            "candidate_selection_primary_metric": "English PadChest-6 macro concept F1",
            "required_metrics": [
                "macro and micro concept F1", "CIDEr-D", "BLEU-4", "ROUGE-L",
                "METEOR", "BERTScore", "validated English contradiction rate",
                "unique generated report fraction", "repeated 4-gram report rate",
                "training-report exact-match rate",
            ],
            "selection_rule": [
                "macro concept F1 at least 0.30",
                "macro concept F1 gain over the frozen v1.1 reference at least 0.05",
                "CIDEr-D at least 0.90",
                "validated contradiction rate at most 0.20",
                "unique generated report fraction at least 0.25",
                "repeated 4-gram report rate at most 0.03",
                "training-report exact-match rate at most 0.05",
            ],
            "all_conditions_required": True,
            "maximum_candidates_advanced": 1,
            "if_no_candidate_passes": (
                "publish the negative v2 development result and do not evaluate the "
                "original Objective 6 locked test"
            ),
            "if_one_candidate_passes": (
                "freeze every parameter, threshold, translator, retriever, prompt, "
                "controller, and metric before one original locked-test evaluation"
            ),
        },
        "privacy_and_safety": {
            "private_reports_public": False,
            "private_manifests_public": False,
            "case_level_outputs_public": False,
            "patient_or_image_identifiers_public": False,
            "medical_images_public": False,
            "private_checkpoints_public": False,
            "aggregate_protocol_and_results_public": True,
            "original_validation_opened_for_v2": False,
            "locked_test_manifest_opened": False,
            "locked_test_reports_accessed": False,
            "locked_test_evaluated": False,
        },
    }
    protocol_path = public / "objective6_english_v2_protocol_public.json"
    protocol_hash = write_json(protocol, protocol_path)
    cohort_summary = {
        "artifact": "Objective 6 English v2 private development-cohort summary",
        "version": VERSION,
        "source_training_manifest_sha256": TRAIN_SHA256,
        "source_cases": len(source),
        "source_patients": int(source["patient_id"].nunique()),
        "v2_training_cases": len(training),
        "v2_training_patients": int(training["patient_id"].nunique()),
        "v2_development_cases": len(development),
        "v2_development_patients": int(development["patient_id"].nunique()),
        "patient_overlap": 0,
        "private_manifest_sha256": private_hashes,
        "patient_or_image_identifiers_public": False,
        "report_content_public": False,
        "translation_performed": False,
        "training_performed": False,
        "original_validation_opened": False,
        "locked_test_manifest_opened": False,
        "locked_test_reports_accessed": False,
        "locked_test_evaluated": False,
    }
    cohort_path = public / "objective6_english_v2_cohort_summary_public.json"
    cohort_hash = write_json(cohort_summary, cohort_path)
    lock = {
        "artifact": "Final Objective 6 English v2 pre-translation protocol lock",
        "version": VERSION,
        "immutable": True,
        "protocol_sha256": protocol_hash,
        "cohort_summary_sha256": cohort_hash,
        "v1_1_summary_sha256": args.expected_v1_1_summary_sha256,
        "v1_1_final_lock_sha256": args.expected_v1_1_lock_sha256,
        "translator_revision": revision,
        "candidate_count": 2,
        "development_evaluation_count": 0,
        "locked_test_evaluation_count": 0,
        "translation_performed": False,
        "v2_training_started": False,
        "original_validation_opened": False,
        "locked_test_manifest_opened": False,
        "locked_test_reports_accessed": False,
        "locked_test_evaluated": False,
    }
    lock_path = public / "FINAL_OBJECTIVE6_ENGLISH_V2_PROTOCOL_LOCK.json"
    lock_hash = write_json(lock, lock_path)

    print("\n--- OBJECTIVE 6 ENGLISH V2 PRIVATE COHORT ---")
    print("Source training cases:", len(source))
    print("v2 training cases:", len(training))
    print("v2 development cases:", len(development))
    print("v2 training patients:", training["patient_id"].nunique())
    print("v2 development patients:", development["patient_id"].nunique())
    print("Patient overlap:", 0)
    print("\n--- FINAL OBJECTIVE 6 ENGLISH V2 LOCK ---")
    print("Protocol SHA-256:", protocol_hash)
    print("Cohort summary SHA-256:", cohort_hash)
    print("Final lock SHA-256:", lock_hash)
    print("Translator revision:", revision)
    print("Translation performed:", False)
    print("Training performed:", False)
    print("Original validation opened:", False)
    print("Locked-test manifest opened:", False)
    print("Locked-test reports accessed:", False)
    print("Locked-test evaluated:", False)
    print("OBJECTIVE 6 ENGLISH V2 PROTOCOL AND PRIVATE DEVELOPMENT COHORT LOCKED SUCCESSFULLY")


if __name__ == "__main__":
    main()
