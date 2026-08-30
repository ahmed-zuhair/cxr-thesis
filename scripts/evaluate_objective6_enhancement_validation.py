#!/usr/bin/env python3
"""Perform the single locked Objective 6 v1.1 validation comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cxr_thesis.objective6.evaluation import (
    cider_document_frequency,
    clinical_scores,
    parse_padchest6_labels,
)
from cxr_thesis.objective6.text import normalise_report, tokenise_report
from evaluate_objective6_validation import (
    label_reports,
    load_system,
    metrics_for_indices,
    percentile_interval,
    point_metrics,
    sha256,
    verify_checksum,
    write_checksum,
    write_json,
)

TRAIN_SHA256 = "278addf3c0a216bb206b4e4b79364f26bacbee977f3209e9275e2abbd8fda7d7"
VAL_SHA256 = "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
PROTOCOL_SHA256 = "279e4fe83da6d82afcbcce595b5596980ca970fae958a16264d3b3e5172eb1a1"
LOCK_SHA256 = "b840440da16023c0169eb3f32c0f4ce7a20ecfa34f8f6b6bfa8ef20511aa53e6"
CHECKPOINT_SHA256 = "bc6c6c27208b31a597890b2f12abc35a7e0979b80d6b77cd5b2e341d43baf89b"
LABELER_SHA256 = "99d6126a3ded1feb749eaf29f3c47a73c0cb323773b44068d0ddf44e4337a731"
ENHANCED = "clinical_guided_multimodal_v1_1"
SYSTEMS = ("image_only", "multimodal", ENHANCED)
EXPECTED_V1 = {
    "image_only": {
        "macro_concept_f1": 0.21871894547689585,
        "CIDEr-D": 0.8056144170926561,
    },
    "multimodal": {
        "macro_concept_f1": 0.20707029780859554,
        "CIDEr-D": 0.8064452646530204,
        "explicit_negation_contradiction_rate": 0.9696969696969697,
        "unique_generated_report_fraction": 0.13280254777070064,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--enhancement-protocol", type=Path, required=True)
    parser.add_argument("--enhancement-lock", type=Path, required=True)
    parser.add_argument("--image-only-root", type=Path, required=True)
    parser.add_argument("--multimodal-root", type=Path, required=True)
    parser.add_argument("--enhanced-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-hf-repo", required=True)
    parser.add_argument("--private-hf-path", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=6142)
    return parser.parse_args()


def load_enhanced_system(root: Path) -> pd.DataFrame:
    inventory_path = root / "private_validation_generation_inventory.json"
    verify_checksum(inventory_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("variant") != ENHANCED
        or inventory.get("cases") != 6280
        or inventory.get("shards") != 20
        or inventory.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or inventory.get("validation_manifest_sha256") != VAL_SHA256
        or inventory.get("enhancement_protocol_sha256") != PROTOCOL_SHA256
        or inventory.get("enhancement_lock_sha256") != LOCK_SHA256
        or inventory.get("test_evaluated") is not False
    ):
        raise RuntimeError("Invalid Objective 6 v1.1 generation inventory")
    frames: list[pd.DataFrame] = []
    expected_start = 0
    for index in range(20):
        directory = root / "shards" / f"shard_{index:03d}"
        predictions = directory / "predictions_private.csv"
        summary_path = directory / "shard_summary_private.json"
        verify_checksum(predictions)
        verify_checksum(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("variant") != ENHANCED
            or summary.get("shard_index") != index
            or summary.get("start_index") != expected_start
            or summary.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or summary.get("predictions_sha256") != sha256(predictions)
            or summary.get("test_evaluated") is not False
        ):
            raise RuntimeError(f"Invalid Objective 6 v1.1 shard {index}")
        frame = pd.read_csv(predictions, low_memory=False).fillna("")
        if len(frame) != int(summary["cases"]):
            raise RuntimeError(f"Objective 6 v1.1 shard row mismatch {index}")
        frames.append(frame)
        expected_start = int(summary["stop_index_exclusive"])
    output = pd.concat(frames, ignore_index=True)
    required = {
        "case_code", "patient_id", "reference_report", "generated_report",
        "reference_labels",
    }
    if (
        len(output) != 6280
        or expected_start != 6280
        or not required.issubset(output.columns)
        or output["case_code"].astype(str).duplicated().any()
    ):
        raise RuntimeError("Invalid Objective 6 v1.1 system output")
    return output


def restore_locked_labeler(
    api: Any,
    repository: str,
    token: str,
    destination: Path,
) -> tuple[Any, list[Any], str]:
    files = api.list_repo_files(repository, repo_type="model", token=token)
    candidates = sorted(
        path for path in files
        if Path(path).name == "padchest6_labeler_private.joblib"
        and path.startswith("objective6/")
    )
    matches: list[tuple[str, Path]] = []
    from huggingface_hub import hf_hub_download

    for remote_path in candidates:
        downloaded = Path(hf_hub_download(
            repo_id=repository, filename=remote_path,
            repo_type="model", token=token,
        ))
        if sha256(downloaded) == LABELER_SHA256:
            matches.append((remote_path, downloaded))
    if not matches:
        raise FileNotFoundError("The frozen Objective 6 clinical labeler is missing")
    if len({sha256(path) for _, path in matches}) != 1:
        raise RuntimeError("Conflicting Objective 6 clinical labelers found")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0][1], destination)
    payload = joblib.load(destination)
    if set(payload) != {"vectorizer", "models"} or len(payload["models"]) != 6:
        raise RuntimeError("Frozen Objective 6 clinical labeler payload changed")
    return payload["vectorizer"], payload["models"], matches[0][0]


def validate_alignment(
    frame: pd.DataFrame,
    validation: pd.DataFrame,
    name: str,
) -> None:
    if frame["case_code"].astype(str).tolist() != validation[
        "case_code"
    ].astype(str).tolist():
        raise RuntimeError(f"Objective 6 {name} case order changed")
    if not (
        frame["reference_report"].map(normalise_report).to_numpy()
        == validation["report"].map(normalise_report).to_numpy()
    ).all():
        raise RuntimeError(f"Objective 6 {name} references changed")
    if not (
        frame["reference_labels"].astype(str).to_numpy()
        == validation["labels"].astype(str).to_numpy()
    ).all():
        raise RuntimeError(f"Objective 6 {name} labels changed")
    if not (
        frame["patient_id"].astype(str).to_numpy()
        == validation["patient_id"].astype(str).to_numpy()
    ).all():
        raise RuntimeError(f"Objective 6 {name} patients changed")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    completed = args.output_dir / "public" / "FINAL_OBJECTIVE6_V1_1_VALIDATION.json"
    if completed.is_file():
        raise RuntimeError("Objective 6 v1.1 validation is already finalized")
    if args.bootstrap_replicates != 1000 or args.bootstrap_seed != 6142:
        raise RuntimeError("Objective 6 v1.1 bootstrap configuration changed")
    protected = {
        args.train_manifest: TRAIN_SHA256,
        args.val_manifest: VAL_SHA256,
        args.enhancement_protocol: PROTOCOL_SHA256,
        args.enhancement_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 v1.1 input changed: {path}")
    protocol = json.loads(args.enhancement_protocol.read_text(encoding="utf-8"))
    lock = json.loads(args.enhancement_lock.read_text(encoding="utf-8"))
    rule = protocol.get("advancement_rule", {})
    if (
        protocol.get("enhanced_architecture", {}).get("name") != ENHANCED
        or protocol.get("validation_evaluation", {}).get("evaluation_count") != 1
        or rule.get("all_conditions_required") is not True
        or rule.get("additional_enhancement_rounds_allowed") is not False
        or lock.get("v1_1_validation_evaluated") is not False
        or lock.get("locked_test_evaluated") is not False
    ):
        raise RuntimeError("Objective 6 v1.1 evaluation lock changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.output_dir / "private"
    public_dir = args.output_dir / "public"
    private_dir.mkdir(exist_ok=True)
    public_dir.mkdir(exist_ok=True)

    train = pd.read_csv(args.train_manifest, low_memory=False).fillna("")
    validation = pd.read_csv(args.val_manifest, low_memory=False).fillna("")
    if len(train) != 29283 or len(validation) != 6280:
        raise RuntimeError("Objective 6 cohort size changed")
    frames = {
        "image_only": load_system(args.image_only_root, "image_only"),
        "multimodal": load_system(args.multimodal_root, "multimodal"),
        ENHANCED: load_enhanced_system(args.enhanced_root),
    }
    for name, frame in frames.items():
        validate_alignment(frame, validation, name)

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    if not bool(api.model_info(args.private_hf_repo, token=token).private):
        raise RuntimeError("Objective 6 recovery repository must remain private")
    labeler_path = private_dir / "padchest6_labeler_private.joblib"
    vectorizer, models, labeler_remote = restore_locked_labeler(
        api, args.private_hf_repo, token, labeler_path
    )
    if sha256(labeler_path) != LABELER_SHA256:
        raise RuntimeError("Frozen Objective 6 clinical labeler changed")

    reference_labels = np.stack(
        validation["labels"].map(parse_padchest6_labels).to_numpy()
    )
    reference_predictions = label_reports(
        vectorizer, models, validation["report"].astype(str).tolist()
    )
    labeler_validation = clinical_scores(reference_labels, reference_predictions)
    reference_tokens = [tokenise_report(value) for value in validation["report"]]
    document_frequency = cider_document_frequency(reference_tokens)
    train_report_set = set(train["report"].map(normalise_report))

    point: dict[str, dict[str, float]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    private_frames = []
    for name in SYSTEMS:
        print("Computing locked Objective 6 v1.1 metrics:", name)
        predicted = label_reports(
            vectorizer, models, frames[name]["generated_report"].astype(str).tolist()
        )
        point[name], arrays[name] = point_metrics(
            frames[name], reference_tokens, document_frequency, train_report_set,
            reference_labels, predicted,
        )
        private_frames.append(pd.DataFrame({
            "case_code": validation["case_code"].astype(str),
            "patient_id": validation["patient_id"].astype(str),
            "system": name,
            "ROUGE-L": arrays[name]["rouge"],
            "METEOR_exact_token": arrays[name]["meteor"],
            "CIDEr-D": arrays[name]["cider"],
            "empty_report": arrays[name]["empty"].astype(int),
            "repeated_4gram": arrays[name]["repeated"].astype(int),
            "training_exact_match": arrays[name]["exact"].astype(int),
            "explicit_contradictions": arrays[name]["contradictions"].astype(int),
            "explicit_mentions": arrays[name]["mentions"].astype(int),
            "reference_labels": ["".join(map(str, row)) for row in reference_labels],
            "predicted_labels": ["".join(map(str, row)) for row in predicted],
        }))

    tolerance = 1e-12
    for system, expected_metrics in EXPECTED_V1.items():
        for metric, expected in expected_metrics.items():
            if abs(float(point[system][metric]) - expected) > tolerance:
                raise RuntimeError(
                    f"Published Objective 6 v1 {system} {metric} did not reproduce"
                )

    patients = validation["patient_id"].astype(str).to_numpy()
    unique_patients = np.unique(patients)
    patient_rows = {
        patient: np.flatnonzero(patients == patient) for patient in unique_patients
    }
    random = np.random.default_rng(args.bootstrap_seed)
    metric_names = tuple(metrics_for_indices(
        np.arange(len(validation)), arrays[SYSTEMS[0]], reference_labels
    ))
    distributions = {
        system: {
            metric: np.zeros(args.bootstrap_replicates) for metric in metric_names
        }
        for system in SYSTEMS
    }
    for replicate in range(args.bootstrap_replicates):
        sampled = random.choice(
            unique_patients, size=len(unique_patients), replace=True
        )
        indices = np.concatenate([patient_rows[patient] for patient in sampled])
        for system in SYSTEMS:
            values = metrics_for_indices(indices, arrays[system], reference_labels)
            for metric, value in values.items():
                distributions[system][metric][replicate] = value
        if (replicate + 1) % 100 == 0:
            print(f"Bootstrap replicate {replicate + 1}/{args.bootstrap_replicates}")

    intervals = {
        system: {
            metric: percentile_interval(values)
            for metric, values in metrics.items()
        }
        for system, metrics in distributions.items()
    }
    paired: dict[str, dict[str, object]] = {}
    for metric in ("macro_concept_f1", "CIDEr-D"):
        difference = (
            distributions[ENHANCED][metric] - distributions["multimodal"][metric]
        )
        paired[metric] = {
            "point_difference": float(
                point[ENHANCED][metric] - point["multimodal"][metric]
            ),
            "bootstrap_95_ci": percentile_interval(difference),
            "two_sided_bootstrap_p": float(min(
                1.0,
                2.0 * min(
                    np.mean(difference <= 0), np.mean(difference >= 0)
                ),
            )),
        }

    enhanced = point[ENHANCED]
    conditions = {
        "minimum_macro_concept_f1": bool(
            enhanced["macro_concept_f1"] >= rule["minimum_macro_concept_f1"]
        ),
        "minimum_absolute_macro_f1_gain_over_v1_multimodal": bool(
            enhanced["macro_concept_f1"] - point["multimodal"]["macro_concept_f1"]
            >= rule["minimum_absolute_macro_f1_gain_over_v1_multimodal"]
        ),
        "minimum_CIDEr_D": bool(enhanced["CIDEr-D"] >= rule["minimum_CIDEr_D"]),
        "maximum_explicit_negation_contradiction_rate": bool(
            enhanced["explicit_negation_contradiction_rate"]
            <= rule["maximum_explicit_negation_contradiction_rate"]
        ),
        "minimum_unique_generated_report_fraction": bool(
            enhanced["unique_generated_report_fraction"]
            >= rule["minimum_unique_generated_report_fraction"]
        ),
        "maximum_repeated_4gram_report_rate": bool(
            enhanced["repeated_4gram_report_rate"]
            <= rule["maximum_repeated_4gram_report_rate"]
        ),
    }
    advances = bool(all(conditions.values()))

    case_metrics = pd.concat(private_frames, ignore_index=True)
    case_path = private_dir / "v1_1_validation_case_metrics_private.csv"
    case_metrics.to_csv(case_path, index=False, lineterminator="\n")
    case_hash = write_checksum(case_path)
    bootstrap_path = private_dir / "v1_1_validation_bootstrap_private.npz"
    np.savez_compressed(
        bootstrap_path,
        **{
            f"{system}__{metric}": values
            for system, metrics in distributions.items()
            for metric, values in metrics.items()
        },
    )
    bootstrap_hash = write_checksum(bootstrap_path)

    summary = {
        "artifact": "Objective 6 locked v1.1 enhancement validation comparison",
        "version": "v1.1.0",
        "validation_cases": len(validation),
        "validation_patients": int(validation["patient_id"].astype(str).nunique()),
        "systems": point,
        "patient_cluster_bootstrap_95_ci": intervals,
        "enhanced_minus_v1_multimodal": paired,
        "advancement_rule": rule,
        "advancement_conditions": conditions,
        "all_conditions_passed": advances,
        "advance_to_single_locked_test_evaluation": advances,
        "additional_enhancement_rounds_allowed": False,
        "clinical_labeler": {
            "bundle_sha256": LABELER_SHA256,
            "remote_source": labeler_remote,
            "validation_reference_report_performance": labeler_validation,
            "validation_or_test_outputs_used_to_fit": False,
        },
        "enhanced_checkpoint_sha256": CHECKPOINT_SHA256,
        "enhancement_protocol_sha256": PROTOCOL_SHA256,
        "enhancement_lock_sha256": LOCK_SHA256,
        "bootstrap": {
            "method": "paired patient-cluster percentile bootstrap",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
        "private_artifact_sha256": {
            "case_metrics": case_hash, "bootstrap": bootstrap_hash,
        },
        "privacy": {
            "raw_reports_public": False,
            "generated_reports_public": False,
            "case_level_metrics_public": False,
            "patient_or_image_identifiers_public": False,
        },
        "validation_evaluation_count": 1,
        "locked_test_manifest_opened": False,
        "locked_test_reports_accessed": False,
        "locked_test_evaluated": False,
    }
    summary_path = public_dir / "objective6_v1_1_validation_summary_public.json"
    summary_hash = write_json(summary, summary_path)

    labels = ["Image-only v1", "Multimodal v1", "Clinical-guided v1.1"]
    colors = ["#3b82f6", "#0f9d76", "#8b5cf6"]
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    panels = (
        ("macro_concept_f1", "Macro concept F1", rule["minimum_macro_concept_f1"]),
        ("CIDEr-D", "CIDEr-D", rule["minimum_CIDEr_D"]),
        (
            "explicit_negation_contradiction_rate",
            "Explicit-negation contradiction rate (lower is better)",
            rule["maximum_explicit_negation_contradiction_rate"],
        ),
        (
            "unique_generated_report_fraction",
            "Unique generated-report fraction",
            rule["minimum_unique_generated_report_fraction"],
        ),
    )
    for axis, (metric, title, threshold) in zip(axes.flat, panels):
        values = [point[system][metric] for system in SYSTEMS]
        axis.bar(labels, values, color=colors)
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Objective 6: Preregistered v1.1 Validation Decision")
    figure.tight_layout()
    figure_path = public_dir / "objective6_v1_1_validation_comparison.png"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    figure_hash = sha256(figure_path)

    final_lock = {
        "artifact": "Final Objective 6 v1.1 validation decision lock",
        "immutable": True,
        "enhancement_protocol_sha256": PROTOCOL_SHA256,
        "enhancement_lock_sha256": LOCK_SHA256,
        "enhanced_checkpoint_sha256": CHECKPOINT_SHA256,
        "summary_sha256": summary_hash,
        "figure_sha256": figure_hash,
        "validation_evaluation_count": 1,
        "all_advancement_conditions_passed": advances,
        "advance_to_single_locked_test_evaluation": advances,
        "additional_enhancement_rounds_allowed": False,
        "locked_test_evaluated": False,
        "locked_test_evaluation_count": 0,
    }
    final_lock_path = public_dir / "FINAL_OBJECTIVE6_V1_1_VALIDATION.json"
    final_lock_hash = write_json(final_lock, final_lock_path)

    remote_root = args.private_hf_path.strip("/")
    upload_files = [
        case_path, case_path.with_suffix(".csv.sha256"),
        bootstrap_path, bootstrap_path.with_suffix(".npz.sha256"),
    ]
    api.create_commit(
        repo_id=args.private_hf_repo, repo_type="model", token=token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{remote_root}/evaluation/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in upload_files
        ],
        commit_message="recovery: Objective 6 private v1.1 validation comparison",
    )

    print("\n--- OBJECTIVE 6 V1.1 LOCKED VALIDATION RESULTS ---")
    for system in SYSTEMS:
        print("SYSTEM:", system)
        for metric in (
            "macro_concept_f1", "CIDEr-D", "BLEU-4", "ROUGE-L",
            "METEOR_exact_token", "explicit_negation_contradiction_rate",
            "repeated_4gram_report_rate", "unique_generated_report_fraction",
        ):
            print(metric + ":", point[system][metric])
    print("\n--- PREREGISTERED ADVANCEMENT DECISION ---")
    for condition, passed in conditions.items():
        print(condition + ":", passed)
    print("All conditions passed:", advances)
    print("Advance to locked test:", advances)
    print("Additional enhancement rounds allowed:", False)
    print("Summary SHA-256:", summary_hash)
    print("Figure SHA-256:", figure_hash)
    print("Final-lock SHA-256:", final_lock_hash)
    print("Private recovery verified:", True)
    print("Raw reports printed:", False)
    print("Case-level outputs publicly uploaded:", False)
    print("Locked-test manifest opened:", False)
    print("Locked-test reports accessed:", False)
    print("Locked-test evaluated:", False)
    print("OBJECTIVE 6 V1.1 LOCKED VALIDATION COMPARISON SUCCESSFUL")


if __name__ == "__main__":
    main()
