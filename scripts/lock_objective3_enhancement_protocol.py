#!/usr/bin/env python3
"""Lock the bounded Objective 3 v1.1 enhancement before any new training."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--expected-original-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not args.original_protocol.is_file():
        raise FileNotFoundError(args.original_protocol)
    if args.output_dir.exists():
        raise FileExistsError(
            "The v1.1 protocol amendment already exists; it must not be overwritten"
        )
    original_hash = sha256_file(args.original_protocol)
    if original_hash != args.expected_original_sha256:
        raise RuntimeError("Original v1.0 protocol SHA-256 does not match")
    original = json.loads(args.original_protocol.read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise RuntimeError("Original protocol must be a JSON object")
    protected_false_fields = (
        "new_test_cohort_selected",
        "test_manifest_opened",
        "test_labels_accessed",
        "test_evaluated",
    )
    failed = [name for name in protected_false_fields if original.get(name) is not False]
    if failed:
        raise RuntimeError(f"Original protocol is not test-blind: {failed}")
    if original.get("quantum_improvement_observed") is not False:
        raise RuntimeError("v1.0 must record that quantum improvement was not observed")

    payload = {
        "artifact": "Objective 3 bounded enhancement protocol amendment",
        "objective": 3,
        "amendment_version": "1.1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes_original_for_future_training": True,
        "original_protocol_preserved": True,
        "original_protocol_sha256": original_hash,
        "reason_for_amendment": (
            "The preregistered v1.0 quantum head did not improve mean validation "
            "AUROC over its parameter-matched classical control. The four-value "
            "branch was concatenated with 160 untouched features and could be diluted."
        ),
        "v1_validation_result": {
            "classical_mean_macro_auroc": 0.6565523408644843,
            "quantum_mean_macro_auroc": 0.6539867052406065,
            "quantum_minus_classical_by_seed": [
                -0.006022755247143574,
                0.0007092287945338338,
                -0.0023833804190236396,
            ],
            "mean_quantum_minus_classical_macro_auroc": -0.0025656356238777933,
        },
        "frozen_enhancement": {
            "architecture": "v1_1_reupload_gated",
            "encoder": "unchanged frozen Objective 2 GAT",
            "embedding_dimension": 160,
            "qubits": 4,
            "data_reuploading_blocks": 3,
            "quantum_bottleneck_parameters": 36,
            "classical_bottleneck_parameters": 36,
            "parameter_matched": True,
            "fusion": "learned gated residual 4-to-160 back-projection",
            "initial_fusion_scale": 0.1,
            "total_trainable_parameters_each": 3253,
            "seeds": [42, 43, 44],
            "training_and_validation_cohorts_unchanged": True,
            "optimization_protocol_unchanged": True,
        },
        "advance_rules": {
            "mean_quantum_minus_classical_validation_macro_auroc_above_zero": True,
            "quantum_seed_wins_required": 2,
            "seed_runs": 3,
            "both_rules_required": True,
            "secondary_metrics": ["macro_auprc", "macro_f1"],
            "secondary_metrics_used_for_selection": False,
            "additional_architecture_tuning_if_rules_fail": False,
        },
        "status": "locked before v1.1 training and before final evaluation",
        "new_test_cohort_selected": False,
        "test_manifest_opened": False,
        "test_labels_accessed": False,
        "test_evaluated": False,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "private_checkpoints_included": False,
        "allowed_for_publication": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = (
        args.output_dir / "objective3_enhancement_protocol_amendment_public.json"
    )
    write_json(protocol_path, payload)
    protocol_hash = sha256_file(protocol_path)
    checksum_path = protocol_path.with_suffix(".json.sha256")
    checksum_path.write_text(
        f"{protocol_hash}  {protocol_path.name}\n", encoding="utf-8"
    )
    print("--- OBJECTIVE 3 V1.1 PROTOCOL AMENDMENT ---")
    print("Original v1.0 SHA-256:", original_hash)
    print("Amendment:", protocol_path)
    print("Amendment SHA-256:", protocol_hash)
    print("Architecture: v1_1_reupload_gated")
    print("Quantum/classical bottleneck parameters: 36/36")
    print("Total trainable parameters per head: 3253")
    print("Test manifest opened: False")
    print("Test labels accessed: False")
    print("Test evaluated: False")
    print("OBJECTIVE 3 V1.1 ENHANCEMENT PROTOCOL LOCK SUCCESSFUL")


if __name__ == "__main__":
    main()
