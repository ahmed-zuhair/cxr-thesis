#!/usr/bin/env python3
"""Lock one factual English-translation remediation before correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_PROTOCOL_SHA256 = "a09241aff9cb998c68023c6399b6bbb33b66b96fd61475a9f0949a42f6143f62"
BASE_LOCK_SHA256 = "28b677450562e04542d4516b2c94cd8b4fa7f9f1161ffe7b992e845391c8d6f4"
TRANSLATION_SUMMARY_SHA256 = "4336e2348f2b735909b78ac1af55235c0440f568b21de75cc062ed0f0b30cd75"
DIAGNOSTIC_SHA256 = "708d3c3a1cbf25ac90153108d5ec639d4b543e8613067bfd8a5596bc26944b9a"
TRANSLATOR_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
VERSION = "v2.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--translation-summary", type=Path, required=True)
    parser.add_argument("--translation-diagnostic", type=Path, required=True)
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
        raise FileExistsError(f"Remediation lock exists: {args.output_dir}")
    protected = {
        args.base_protocol: BASE_PROTOCOL_SHA256,
        args.base_lock: BASE_LOCK_SHA256,
        args.translation_summary: TRANSLATION_SUMMARY_SHA256,
        args.translation_diagnostic: DIAGNOSTIC_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 English v2 input changed: {path}")

    base_protocol = json.loads(args.base_protocol.read_text(encoding="utf-8"))
    base_lock = json.loads(args.base_lock.read_text(encoding="utf-8"))
    translation = json.loads(args.translation_summary.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.translation_diagnostic.read_text(encoding="utf-8"))
    if (
        base_protocol.get("version") != "v2.0.0"
        or base_protocol.get("v2_candidates", {}).get("candidate_count") != 2
        or base_lock.get("immutable") is not True
        or base_lock.get("translator_revision") != TRANSLATOR_REVISION
        or translation.get("translation_quality_gate_passed") is not False
        or translation.get("locked_test_evaluated") is not False
        or diagnostic.get("diagnostic_only") is not True
        or diagnostic.get("candidate_training_started") is not False
        or diagnostic.get("candidate_generation_started") is not False
        or diagnostic.get("locked_test_evaluated") is not False
        or diagnostic.get("raw_reports_printed") is not False
    ):
        raise RuntimeError("Objective 6 English v2 remediation preconditions changed")

    initial = {
        "unique_source_reports": int(diagnostic["unique_source_reports"]),
        "nonempty_fraction": float(translation["nonempty_fraction"]),
        "number_or_measurement_loss_rate": float(
            diagnostic["number_or_measurement_loss_rate"]
        ),
        "clause_aware_concept_lexical_coverage": float(
            diagnostic["clause_aware_concept_lexical_coverage"]
        ),
        "clause_aware_concept_polarity_agreement": float(
            diagnostic["clause_aware_concept_polarity_agreement"]
        ),
        "English_marker_dominant_reports": int(
            diagnostic["target_language_marker_diagnostics"][
                "english_marker_dominant"
            ]
        ),
        "Spanish_marker_dominant_reports": int(
            diagnostic["target_language_marker_diagnostics"][
                "spanish_marker_dominant"
            ]
        ),
        "marker_tied_or_free_reports": int(
            diagnostic["target_language_marker_diagnostics"][
                "tied_or_marker_free"
            ]
        ),
    }
    if (
        initial["unique_source_reports"] != 18984
        or initial["nonempty_fraction"] != 1.0
        or initial["number_or_measurement_loss_rate"] <= 0.01
        or initial["clause_aware_concept_polarity_agreement"] >= 0.95
        or initial["Spanish_marker_dominant_reports"] != 2070
    ):
        raise RuntimeError("The documented Objective 6 translation failure changed")

    public = args.output_dir / "public"
    public.mkdir(parents=True)
    protocol = {
        "artifact": "Objective 6 English v2 factual-translation remediation protocol",
        "version": VERSION,
        "status": (
            "locked after the failed v2.0.0 translation-quality gate and before "
            "any report correction, enhancement training, candidate generation, "
            "development evaluation, or locked-test access"
        ),
        "repository_commit_before_remediation": args.repository_commit,
        "protected_predecessors": {
            "base_v2_protocol_sha256": BASE_PROTOCOL_SHA256,
            "base_v2_final_lock_sha256": BASE_LOCK_SHA256,
            "failed_translation_summary_sha256": TRANSLATION_SUMMARY_SHA256,
            "aggregate_diagnostic_sha256": DIAGNOSTIC_SHA256,
            "failed_translation_result_retained": True,
            "failed_translation_metrics_rewritten": False,
        },
        "aggregate_failure_evidence": initial,
        "scientific_scope": {
            "purpose": (
                "repair reference-language fidelity before either preregistered "
                "enhancement candidate is fitted"
            ),
            "remediation_attempts_allowed": 1,
            "candidate_count_changed": False,
            "candidate_architectures_changed": False,
            "candidate_selection_thresholds_changed": False,
            "development_patients_changed": False,
            "original_validation_used": False,
            "locked_test_used": False,
        },
        "locked_remediation": {
            "translator": "facebook/nllb-200-distilled-600M",
            "translator_revision": TRANSLATOR_REVISION,
            "source_language": "Spanish",
            "target_language": "English",
            "numeric_measurement_shield": (
                "replace every source numeric token with an indexed sentinel before "
                "translation and restore the exact normalized numeric token afterward"
            ),
            "canonical_clinical_terminology_controller": {
                "targets": [
                    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                    "Effusion", "Pneumothorax",
                ],
                "operation": (
                    "for concepts explicitly stated in the Spanish source only, "
                    "remove contradictory translated concept clauses and append one "
                    "canonical English sentence with the same positive or negative "
                    "polarity; structured labels never create reference statements"
                ),
            },
            "English_language_enforcement": (
                "apply one additional frozen-NLLB pass to Spanish-marker-dominant "
                "translations, then reapply numeric and terminology preservation"
            ),
            "normalisation": "Unicode NFC and whitespace normalization only",
            "manual_report_editing": False,
            "external_translation_API": False,
            "source_structured_labels_used_to_invent_report_content": False,
        },
        "locked_quality_gates": {
            "minimum_nonempty_fraction": 0.995,
            "maximum_number_or_measurement_loss_rate": 0.01,
            "minimum_PadChest6_concept_polarity_agreement": 0.95,
            "maximum_Spanish_marker_dominant_fraction": 0.0,
            "minimum_source_hash_alignment_fraction": 1.0,
            "all_conditions_required": True,
        },
        "decision_rule": {
            "if_all_gates_pass": (
                "publish only aggregate remediation quality, freeze the English "
                "references privately, then begin both original v2 candidates"
            ),
            "if_any_gate_fails": (
                "publish the negative remediation result and stop Objective 6 v2 "
                "without enhancement training or locked-test evaluation"
            ),
        },
        "enhancement_commitment": {
            "translation_is_only_input_preparation": True,
            "candidate_1": "fact_aware_retrieve_edit_english",
            "candidate_2": "fact_aware_retrieval_augmented_english_generator",
            "both_candidates_will_be_developed_if_remediation_passes": True,
        },
        "privacy_and_safety": {
            "raw_Spanish_reports_public": False,
            "raw_English_reports_public": False,
            "private_manifests_public": False,
            "patient_or_image_identifiers_public": False,
            "case_level_outputs_public": False,
            "medical_images_public": False,
            "private_checkpoints_public": False,
            "enhancement_training_started": False,
            "candidate_generation_started": False,
            "original_validation_opened": False,
            "locked_test_manifest_opened": False,
            "locked_test_reports_accessed": False,
            "locked_test_evaluated": False,
        },
    }
    protocol_path = public / "objective6_english_v2_remediation_protocol_public.json"
    protocol_hash = write_json(protocol, protocol_path)
    lock = {
        "artifact": "Final Objective 6 English v2 factual-remediation lock",
        "version": VERSION,
        "immutable": True,
        "protocol_sha256": protocol_hash,
        "base_v2_protocol_sha256": BASE_PROTOCOL_SHA256,
        "failed_translation_summary_sha256": TRANSLATION_SUMMARY_SHA256,
        "aggregate_diagnostic_sha256": DIAGNOSTIC_SHA256,
        "translator_revision": TRANSLATOR_REVISION,
        "remediation_attempts_allowed": 1,
        "remediation_attempts_completed": 0,
        "candidate_count": 2,
        "enhancement_training_started": False,
        "candidate_generation_started": False,
        "development_evaluation_count": 0,
        "original_validation_opened": False,
        "locked_test_manifest_opened": False,
        "locked_test_reports_accessed": False,
        "locked_test_evaluated": False,
    }
    lock_path = public / "FINAL_OBJECTIVE6_ENGLISH_V2_REMEDIATION_LOCK.json"
    lock_hash = write_json(lock, lock_path)
    print("--- OBJECTIVE 6 ENGLISH V2 REMEDIATION LOCK ---")
    print("Protocol SHA-256:", protocol_hash)
    print("Final-lock SHA-256:", lock_hash)
    print("Remediation attempts allowed:", 1)
    print("Enhancement candidate count:", 2)
    print("Translation correction performed:", False)
    print("Enhancement training started:", False)
    print("Original validation opened:", False)
    print("Locked-test evaluated:", False)
    print("OBJECTIVE 6 ENGLISH V2 FACTUAL REMEDIATION LOCKED SUCCESSFULLY")


if __name__ == "__main__":
    main()
