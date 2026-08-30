#!/usr/bin/env python3
"""Lock the single Objective 6 v1.1 enhancement round before implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

V1_SUMMARY_SHA256 = "0bec4540a38993e23327cde334a6a73f97c0a10d4d7d736b6e9f82afa86bcc7a"
V1_LOCK_SHA256 = "a35f55328480be74f01ed8f0879796e82792b1b8946464c4a0976892727d031f"
TRAIN_SHA256 = "278addf3c0a216bb206b4e4b79364f26bacbee977f3209e9275e2abbd8fda7d7"
VAL_SHA256 = "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
V1_MULTIMODAL_SHA256 = "18aa4293195b77aaf04df1ba310431df83f75e51ee6aa5837ff43a48a8ec10d3"
LABELER_SHA256 = "99d6126a3ded1feb749eaf29f3c47a73c0cb323773b44068d0ddf44e4337a731"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-summary", type=Path, required=True)
    parser.add_argument("--v1-lock", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--v1-multimodal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
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


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Objective 6 v1.1 enhancement protocol already exists")
    protected = {
        args.v1_summary: V1_SUMMARY_SHA256,
        args.v1_lock: V1_LOCK_SHA256,
        args.train_manifest: TRAIN_SHA256,
        args.val_manifest: VAL_SHA256,
        args.v1_multimodal_checkpoint: V1_MULTIMODAL_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 input changed: {path}")
    summary = json.loads(args.v1_summary.read_text(encoding="utf-8"))
    final_lock = json.loads(args.v1_lock.read_text(encoding="utf-8"))
    multimodal = summary.get("systems", {}).get("multimodal", {})
    image_only = summary.get("systems", {}).get("image_only", {})
    labeler = summary.get("clinical_labeler", {})
    if (
        summary.get("primary_system") != "multimodal"
        or summary.get("primary_system_changed_after_validation") is not False
        or summary.get("locked_test_evaluated") is not False
        or final_lock.get("locked_test_evaluated") is not False
        or final_lock.get("locked_test_evaluation_count") != 0
        or labeler.get("bundle_sha256") != LABELER_SHA256
    ):
        raise RuntimeError("Objective 6 v1 validation state changed")
    expected_values = {
        "multimodal_macro_f1": 0.20707029780859554,
        "image_only_macro_f1": 0.21871894547689585,
        "multimodal_cider": 0.8064452646530204,
        "multimodal_contradiction": 0.9696969696969697,
        "multimodal_unique": 0.13280254777070064,
        "labeler_reference_macro_f1": 0.8417787544734954,
    }
    actual_values = {
        "multimodal_macro_f1": multimodal.get("macro_concept_f1"),
        "image_only_macro_f1": image_only.get("macro_concept_f1"),
        "multimodal_cider": multimodal.get("CIDEr-D"),
        "multimodal_contradiction": multimodal.get(
            "explicit_negation_contradiction_rate"
        ),
        "multimodal_unique": multimodal.get("unique_generated_report_fraction"),
        "labeler_reference_macro_f1": labeler.get(
            "validation_reference_report_performance", {}
        ).get("macro_concept_f1"),
    }
    if actual_values != expected_values:
        raise RuntimeError("Objective 6 v1 validation values changed")

    args.output_dir.mkdir(parents=True)
    protocol = {
        "artifact": "Objective 6 clinical-guided report-generation enhancement protocol",
        "version": "v1.1.0",
        "status": (
            "locked after publication of v1 validation results and before v1.1 "
            "implementation, training, generation, or locked-test access"
        ),
        "repository_commit_before_implementation": args.repository_commit,
        "motivation": {
            "v1_validation_summary_sha256": V1_SUMMARY_SHA256,
            "v1_validation_lock_sha256": V1_LOCK_SHA256,
            "v1_multimodal_macro_concept_f1": expected_values[
                "multimodal_macro_f1"
            ],
            "v1_image_only_macro_concept_f1": expected_values[
                "image_only_macro_f1"
            ],
            "v1_multimodal_CIDEr_D": expected_values["multimodal_cider"],
            "v1_multimodal_explicit_negation_contradiction_rate": (
                expected_values["multimodal_contradiction"]
            ),
            "v1_multimodal_unique_generated_report_fraction": expected_values[
                "multimodal_unique"
            ],
            "training_only_labeler_validation_reference_macro_f1": (
                expected_values["labeler_reference_macro_f1"]
            ),
            "interpretation": (
                "v1 produced limited clinical efficacy, very high explicit "
                "contradiction, and low report diversity; v1.1 is one transparent "
                "validation-developed enhancement round"
            ),
        },
        "protected_inputs": {
            "training_manifest_sha256": TRAIN_SHA256,
            "validation_manifest_sha256": VAL_SHA256,
            "v1_multimodal_checkpoint_sha256": V1_MULTIMODAL_SHA256,
            "frozen_training_only_clinical_labeler_sha256": LABELER_SHA256,
            "training_cases": 29283,
            "validation_cases": 6280,
            "locked_test_cases_accessed": 0,
        },
        "enhanced_architecture": {
            "name": "clinical_guided_multimodal_v1_1",
            "initialization": "published v1 multimodal checkpoint",
            "visual_encoder": "Objective 5 PadChest-adapted DenseNet-121",
            "visual_tokens": "full final spatial feature map",
            "trainable_encoder_modules": ["denseblock4", "norm5"],
            "frozen_encoder_modules": "all DenseNet modules except denseblock4 and norm5",
            "batch_normalization_mode": "evaluation statistics remain frozen",
            "clinical_context": "age, sex, and projection view token",
            "clinical_concept_head": {
                "concepts": [
                    "Atelectasis", "Cardiomegaly", "Consolidation",
                    "Edema", "Effusion", "Pneumothorax",
                ],
                "targets": "training-manifest PadChest labels only",
                "loss": "class-weighted binary cross-entropy with logits",
                "positive_weights": "calculated from training cases only",
                "concept_token": (
                    "sigmoid concept probabilities projected to one decoder-memory token"
                ),
                "ground_truth_labels_supplied_during_generation": False,
            },
            "decoder": {
                "type": "autoregressive Transformer",
                "d_model": 256,
                "attention_heads": 8,
                "layers": 4,
                "feedforward_dimension": 1024,
                "dropout": 0.1,
                "maximum_tokens_including_bos_eos": 160,
                "vocabulary": "unchanged v1 training-only vocabulary",
            },
        },
        "optimization": {
            "seed": 42,
            "maximum_epochs": 12,
            "early_stopping_patience": 4,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "automatic_mixed_precision": True,
            "gradient_norm_clip": 1.0,
            "optimizer": "AdamW",
            "new_and_decoder_parameter_learning_rate": 0.0001,
            "unfrozen_encoder_learning_rate": 0.00002,
            "weight_decay": 0.0001,
            "scheduler": "ReduceLROnPlateau factor 0.5 patience 2",
            "loss": {
                "report_cross_entropy_weight": 1.0,
                "clinical_binary_cross_entropy_weight": 0.35,
                "adjacent_token_repetition_penalty_weight": 0.02,
                "report_label_smoothing": 0.05,
            },
            "checkpoint_selection": "minimum validation total joint loss",
            "private_epoch_recovery_required": True,
        },
        "decoding": {
            "method": "deterministic beam search",
            "beam_width": 3,
            "length_normalization_alpha": 0.7,
            "no_repeat_ngram_size": 4,
            "maximum_tokens_including_bos_eos": 160,
            "sampling": False,
            "validation_generation_repetitions": 1,
        },
        "validation_evaluation": {
            "evaluation_count": 1,
            "same_6280_case_validation_cohort": True,
            "same_frozen_training_only_labeler": True,
            "same_tokenization_and_metric_implementations": True,
            "comparators": ["v1 image_only", "v1 multimodal", "v1.1 enhanced"],
            "primary_metric": "PadChest-6 macro concept F1",
            "secondary_metric": "CIDEr-D",
            "required_metrics": [
                "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L",
                "Spanish exact-token METEOR", "CIDEr-D",
                "micro concept precision", "micro concept recall",
                "micro concept F1", "macro concept F1",
                "explicit-negation contradiction rate", "empty-report rate",
                "report-with-repeated-4-gram rate",
                "training-report exact-match rate",
                "unique-generated-report fraction",
            ],
            "uncertainty": {
                "method": "paired patient-cluster percentile bootstrap",
                "replicates": 1000,
                "seed": 6142,
                "confidence_level": 0.95,
            },
        },
        "advancement_rule": {
            "all_conditions_required": True,
            "minimum_macro_concept_f1": 0.22707029780859555,
            "minimum_absolute_macro_f1_gain_over_v1_multimodal": 0.02,
            "minimum_CIDEr_D": 0.7864452646530204,
            "maximum_CIDEr_D_drop_from_v1_multimodal": 0.02,
            "maximum_explicit_negation_contradiction_rate": 0.85,
            "minimum_unique_generated_report_fraction": 0.20,
            "maximum_repeated_4gram_report_rate": 0.03,
            "if_passed": (
                "freeze v1.1 as the prospectively advanced enhancement candidate; "
                "publish its validation result and lock one final test comparison"
            ),
            "if_failed": (
                "publish the negative v1.1 result, perform no further Objective 6 "
                "architecture tuning, and exclude v1.1 from confirmatory test claims"
            ),
            "additional_enhancement_rounds_allowed": False,
        },
        "scientific_status": {
            "v1_result_retained": True,
            "v1_primary_system_rewritten": False,
            "v1_1_is_validation_developed": True,
            "final_claim_requires_untouched_locked_test": True,
        },
        "privacy": {
            "raw_reports_public": False,
            "generated_reports_public": False,
            "case_level_metrics_public": False,
            "patient_or_image_identifiers_public": False,
            "medical_images_public": False,
            "private_manifests_public": False,
            "private_checkpoints_public": False,
            "aggregate_protocol_and_results_public": True,
        },
        "safety_state": {
            "v1_1_implementation_started": False,
            "v1_1_training_started": False,
            "v1_1_validation_generation_started": False,
            "locked_test_manifest_opened": False,
            "locked_test_reports_accessed": False,
            "locked_test_evaluated": False,
        },
    }
    protocol_path = args.output_dir / "objective6_enhancement_protocol_public.json"
    protocol_hash = write_json(protocol, protocol_path)
    lock = {
        "artifact": "Final Objective 6 v1.1 pre-implementation enhancement lock",
        "immutable": True,
        "protocol_sha256": protocol_hash,
        "v1_validation_summary_sha256": V1_SUMMARY_SHA256,
        "v1_validation_lock_sha256": V1_LOCK_SHA256,
        "enhancement_rounds_allowed": 1,
        "enhancement_rounds_completed": 0,
        "v1_1_implementation_started": False,
        "v1_1_training_started": False,
        "v1_1_validation_evaluated": False,
        "locked_test_evaluated": False,
        "locked_test_evaluation_count": 0,
    }
    lock_path = args.output_dir / "FINAL_OBJECTIVE6_ENHANCEMENT_PROTOCOL_LOCK.json"
    lock_hash = write_json(lock, lock_path)
    print(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False))
    print("\n--- FINAL OBJECTIVE 6 V1.1 ENHANCEMENT LOCK ---")
    print("Protocol SHA-256:", protocol_hash)
    print("Final-lock SHA-256:", lock_hash)
    print("Enhancement rounds allowed:", 1)
    print("Enhancement training started:", False)
    print("Locked-test manifest opened:", False)
    print("Locked-test reports accessed:", False)
    print("Locked-test evaluated:", False)
    print("OBJECTIVE 6 V1.1 ENHANCEMENT PROTOCOL LOCKED SUCCESSFULLY")


if __name__ == "__main__":
    main()
