#!/usr/bin/env python3
"""Lock Objective 6 validation generation and comparison before inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PRETRAINING_PROTOCOL_SHA256 = (
    "1afed66a5d1cde28acd4271e3a102bdaea38228bb96dbe826395c0a3030b83f3"
)
PRETRAINING_LOCK_SHA256 = (
    "9c66e494f3247aa00782133d671e004ded0a28d4a9626491279ff65d36b0aa6d"
)
TRAIN_MANIFEST_SHA256 = (
    "278addf3c0a216bb206b4e4b79364f26bacbee977f3209e9275e2abbd8fda7d7"
)
VALIDATION_MANIFEST_SHA256 = (
    "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
)
SOURCE_CHECKPOINT_SHA256 = (
    "109db89a723c6e2f24442cb5866bfcf4084e85083936cda91bce3b8ae4365d9d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-protocol", type=Path, required=True)
    parser.add_argument("--pretraining-lock", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--image-only-output", type=Path, required=True)
    parser.add_argument("--multimodal-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def write_json_with_checksum(payload: dict[str, Any], path: Path) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def validate_candidate(directory: Path, variant: str) -> dict[str, Any]:
    required = (
        "best.pt", "best.pt.sha256", "vocabulary.json",
        "vocabulary.json.sha256", "validation_summary.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {variant} artifacts: {missing}")
    checkpoint = directory / "best.pt"
    vocabulary = directory / "vocabulary.json"
    summary = read_json(directory / "validation_summary.json")
    checkpoint_hash = sha256(checkpoint)
    recorded_checkpoint = (directory / "best.pt.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    vocabulary_hash = sha256(vocabulary)
    recorded_vocabulary = (directory / "vocabulary.json.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    checks = {
        "variant": summary.get("variant") == variant,
        "research_result": summary.get("research_result") is True,
        "training_cases": summary.get("training_cases") == 29283,
        "validation_cases": summary.get("validation_cases") == 6280,
        "test_cases_accessed": summary.get("test_cases_accessed") == 0,
        "test_evaluated": summary.get("test_evaluated") is False,
        "source_checkpoint": summary.get("source_checkpoint_sha256")
        == SOURCE_CHECKPOINT_SHA256,
        "checkpoint_recorded": checkpoint_hash == recorded_checkpoint,
        "checkpoint_summary": checkpoint_hash == summary.get("checkpoint_sha256"),
        "vocabulary_recorded": vocabulary_hash == recorded_vocabulary,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"Invalid {variant} validation candidate: {failures}")
    return {
        "variant": variant,
        "checkpoint_sha256": checkpoint_hash,
        "vocabulary_sha256": vocabulary_hash,
        "best_epoch": int(summary["best_epoch"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "validation_loss": float(summary["validation_loss"]),
        "validation_perplexity": float(summary["validation_perplexity"]),
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Objective 6 validation-evaluation lock already exists")
    protected = {
        args.pretraining_protocol: PRETRAINING_PROTOCOL_SHA256,
        args.pretraining_lock: PRETRAINING_LOCK_SHA256,
        args.train_manifest: TRAIN_MANIFEST_SHA256,
        args.val_manifest: VALIDATION_MANIFEST_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 input changed: {path}")
    pretraining = read_json(args.pretraining_protocol)
    pretraining_lock = read_json(args.pretraining_lock)
    if pretraining.get("status") != (
        "locked before cohort materialisation, vocabulary fitting, training, and evaluation"
    ):
        raise RuntimeError("Objective 6 pretraining protocol status changed")
    if pretraining_lock.get("locked_test_evaluated") is not False:
        raise RuntimeError("Objective 6 locked test was already evaluated")

    image_only = validate_candidate(args.image_only_output, "image_only")
    multimodal = validate_candidate(args.multimodal_output, "multimodal")
    if image_only["vocabulary_sha256"] != multimodal["vocabulary_sha256"]:
        raise RuntimeError("Objective 6 candidates do not share one training vocabulary")

    args.output_dir.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "artifact": "Objective 6 validation generation and comparison protocol",
        "version": "v1.0.0",
        "status": (
            "locked after validation-loss training and before full validation "
            "generation or locked-test access"
        ),
        "repository_commit": args.repository_commit,
        "pretraining_protocol_sha256": PRETRAINING_PROTOCOL_SHA256,
        "pretraining_lock_sha256": PRETRAINING_LOCK_SHA256,
        "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "candidate_training_results": {
            "image_only": image_only,
            "multimodal": multimodal,
        },
        "primary_system": {
            "variant": "multimodal",
            "basis": "fixed by the published pretraining protocol",
            "selected_after_generation_metrics": False,
        },
        "comparison_systems": [
            "nearest-training-image report retrieval",
            "image_only",
            "multimodal",
        ],
        "decoding": {
            "method": "deterministic greedy autoregressive decoding",
            "maximum_tokens_including_bos_eos": 160,
            "beam_search": False,
            "sampling": False,
            "temperature_tuning": False,
            "validation_generation_repetitions": 1,
        },
        "retrieval_baseline": {
            "encoder": "frozen Objective 5 PadChest-adapted DenseNet-121",
            "representation": "global-average-pooled final convolutional features",
            "similarity": "cosine",
            "candidate_reports": "training cohort only",
            "tie_break": "lexicographic private case_code",
            "labels_used": False,
            "report_content_used_to_choose_neighbor": False,
        },
        "clinical_efficacy": {
            "name": "PadChest-6 clinical efficacy",
            "concepts": [
                "Atelectasis", "Cardiomegaly", "Consolidation",
                "Edema", "Effusion", "Pneumothorax",
            ],
            "reference": "private PadChest metadata labels",
            "generated_report_labeler": {
                "model": "six independent logistic regressions",
                "features": "Spanish character TF-IDF word-boundary n-grams 3-5",
                "training_text": "Objective 6 training reports only",
                "training_targets": "the six fixed PadChest concepts",
                "probability_threshold": 0.5,
                "validation_or_test_outputs_used_to_fit_labeler": False,
            },
            "metrics": [
                "micro concept precision", "micro concept recall",
                "micro concept F1", "macro concept F1",
                "explicit-negation contradiction rate",
            ],
            "primary_metric": "macro concept F1",
        },
        "lexical_metrics": {
            "reported": [
                "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                "ROUGE-L", "METEOR", "CIDEr-D",
            ],
            "secondary_metric": "CIDEr-D",
            "tokenisation": "locked Objective 6 Unicode Spanish tokenizer",
            "case_handling": "NFKC-normalised lowercase",
            "one_reference_per_study": True,
        },
        "safety_metrics": {
            "reported": [
                "empty-report rate",
                "report-with-repeated-4-gram rate",
                "training-report exact-match rate",
                "unique-generated-report fraction",
            ],
            "empty_after_special_token_removal": True,
        },
        "interpretation_rule": {
            "primary_claim": (
                "multimodal minus image_only macro concept F1 on validation"
            ),
            "secondary_claim": "multimodal minus image_only CIDEr-D on validation",
            "primary_model_is_not_changed_by_validation_result": True,
            "all_preregistered_metrics_reported": True,
        },
        "uncertainty": {
            "method": "paired patient-cluster percentile bootstrap",
            "replicates": 1000,
            "seed": 6042,
            "confidence_level": 0.95,
            "same_resamples_for_all_systems": True,
        },
        "privacy": {
            "raw_reference_reports_public": False,
            "generated_reports_public": False,
            "case_level_metrics_public": False,
            "patient_or_image_identifiers_public": False,
            "medical_images_public": False,
            "private_checkpoints_public": False,
            "aggregate_metrics_public": True,
        },
        "safety_state": {
            "full_validation_generation_started": False,
            "locked_test_manifest_opened": False,
            "locked_test_reports_accessed": False,
            "locked_test_evaluated": False,
        },
    }
    protocol_path = args.output_dir / "objective6_validation_evaluation_protocol_public.json"
    protocol_hash = write_json_with_checksum(protocol, protocol_path)
    lock = {
        "artifact": "Final Objective 6 pre-validation-generation lock",
        "immutable": True,
        "protocol_sha256": protocol_hash,
        "candidate_checkpoint_sha256": {
            "image_only": image_only["checkpoint_sha256"],
            "multimodal": multimodal["checkpoint_sha256"],
        },
        "primary_system": "multimodal",
        "full_validation_generation_started": False,
        "locked_test_evaluated": False,
        "locked_test_evaluation_count": 0,
    }
    lock_path = args.output_dir / "FINAL_OBJECTIVE6_VALIDATION_EVALUATION_LOCK.json"
    lock_hash = write_json_with_checksum(lock, lock_path)
    print(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False))
    print("\n--- FINAL OBJECTIVE 6 VALIDATION-EVALUATION LOCK ---")
    print("Protocol SHA-256:", protocol_hash)
    print("Final-lock SHA-256:", lock_hash)
    print("Primary system fixed before generation:", True)
    print("Full validation generation started:", False)
    print("Locked-test manifest opened:", False)
    print("Locked-test evaluated:", False)
    print("OBJECTIVE 6 VALIDATION GENERATION AND COMPARISON PROTOCOL LOCKED")


if __name__ == "__main__":
    main()
