#!/usr/bin/env python3
"""Evaluate the three locked Objective 6 systems on validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from cxr_thesis.objective6.evaluation import (
    PAD_CHEST_6,
    bleu_statistics,
    cider_d_score,
    cider_document_frequency,
    clinical_scores,
    corpus_bleu,
    exact_token_meteor,
    explicit_contradictions,
    parse_padchest6_labels,
    repeated_ngram,
    rouge_l_f1,
)
from cxr_thesis.objective6.text import normalise_report, tokenise_report

TRAIN_SHA256 = "278addf3c0a216bb206b4e4b79364f26bacbee977f3209e9275e2abbd8fda7d7"
VAL_SHA256 = "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
PROTOCOL_SHA256 = "81424c30f1619707325f0a83ef9a6fba3a859743e3b4ee0c33ac68dba6161438"
LOCK_SHA256 = "e48b11cc0af8be0866b873ae91dd5f4c55738b39927d6dec52d2f29cf5f8275a"
SYSTEMS = (
    "nearest_training_image_retrieval",
    "image_only",
    "multimodal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--validation-protocol", type=Path, required=True)
    parser.add_argument("--validation-lock", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--image-only-root", type=Path, required=True)
    parser.add_argument("--multimodal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-hf-repo", required=True)
    parser.add_argument("--private-hf-path", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=6042)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksum(path: Path) -> str:
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def write_json(payload: dict[str, Any], path: Path) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return write_checksum(path)


def verify_checksum(path: Path) -> None:
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not checksum.is_file() or sha256(path) != checksum.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError(f"Checksum mismatch: {path}")


def load_system(root: Path, variant: str, expected_cases: int = 6280) -> pd.DataFrame:
    inventory_names = (
        "private_validation_retrieval_inventory.json",
        "private_validation_generation_inventory.json",
    )
    inventory_paths = [root / name for name in inventory_names if (root / name).is_file()]
    if len(inventory_paths) != 1:
        raise RuntimeError(f"Expected one private inventory under {root}")
    inventory_path = inventory_paths[0]
    verify_checksum(inventory_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("variant") != variant
        or int(inventory.get("cases", -1)) != expected_cases
        or inventory.get("test_evaluated") is not False
        or inventory.get("validation_protocol_sha256") != PROTOCOL_SHA256
        or inventory.get("validation_lock_sha256") != LOCK_SHA256
    ):
        raise RuntimeError(f"Invalid Objective 6 inventory: {inventory_path}")
    shard_count = int(inventory["shards"])
    frames: list[pd.DataFrame] = []
    expected_start = 0
    for index in range(shard_count):
        directory = root / "shards" / f"shard_{index:03d}"
        predictions = directory / "predictions_private.csv"
        summary_path = directory / "shard_summary_private.json"
        for path in (predictions, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            verify_checksum(path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("variant") != variant
            or int(summary.get("shard_index", -1)) != index
            or int(summary.get("start_index", -1)) != expected_start
            or summary.get("predictions_sha256") != sha256(predictions)
            or summary.get("test_evaluated") is not False
        ):
            raise RuntimeError(f"Invalid Objective 6 shard: {summary_path}")
        frame = pd.read_csv(predictions, low_memory=False).fillna("")
        if len(frame) != int(summary["cases"]):
            raise RuntimeError(f"Objective 6 shard row mismatch: {predictions}")
        frames.append(frame)
        expected_start = int(summary["stop_index_exclusive"])
    output = pd.concat(frames, ignore_index=True)
    required = {
        "case_code", "patient_id", "reference_report", "generated_report", "reference_labels"
    }
    if len(output) != expected_cases or not required.issubset(output.columns):
        raise RuntimeError(f"Invalid Objective 6 system output: {root}")
    if output["case_code"].astype(str).duplicated().any():
        raise RuntimeError(f"Duplicate Objective 6 cases: {root}")
    return output


def fit_or_restore_labeler(
    train: pd.DataFrame,
    output_dir: Path,
    api: Any,
    repository: str,
    remote_root: str,
    token: str,
) -> tuple[Any, list[Any], str, str]:
    bundle_path = output_dir / "padchest6_labeler_private.joblib"
    metadata_path = output_dir / "padchest6_labeler_metadata_private.json"
    remote_bundle = f"{remote_root}/labeler/{bundle_path.name}"
    remote_metadata = f"{remote_root}/labeler/{metadata_path.name}"
    files = set(api.list_repo_files(repository, repo_type="model", token=token))
    restored = remote_bundle in files and remote_metadata in files
    if restored:
        from huggingface_hub import hf_hub_download

        bundle_download = Path(hf_hub_download(
            repository, remote_bundle, repo_type="model", token=token,
            local_dir=str(output_dir / "hf_restore"),
        ))
        metadata_download = Path(hf_hub_download(
            repository, remote_metadata, repo_type="model", token=token,
            local_dir=str(output_dir / "hf_restore"),
        ))
        bundle_path.write_bytes(bundle_download.read_bytes())
        metadata_path.write_bytes(metadata_download.read_bytes())
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("train_manifest_sha256") != TRAIN_SHA256
            or metadata.get("concepts") != list(PAD_CHEST_6)
            or metadata.get("bundle_sha256") != sha256(bundle_path)
        ):
            raise RuntimeError("Private Objective 6 clinical labeler recovery changed")
        payload = joblib.load(bundle_path)
        return payload["vectorizer"], payload["models"], sha256(bundle_path), "restored"

    reports = train["report"].map(normalise_report).tolist()
    targets = np.stack(train["labels"].map(parse_padchest6_labels).to_numpy())
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=50000,
        sublinear_tf=True, dtype=np.float32, lowercase=False,
    )
    features = vectorizer.fit_transform(reports)
    models = []
    for column, label in enumerate(PAD_CHEST_6):
        if np.unique(targets[:, column]).size != 2:
            raise RuntimeError(f"Training-only labeler target is degenerate: {label}")
        model = LogisticRegression(
            class_weight="balanced", max_iter=500, solver="liblinear",
            random_state=42,
        )
        model.fit(features, targets[:, column])
        models.append(model)
    joblib.dump({"vectorizer": vectorizer, "models": models}, bundle_path, compress=3)
    bundle_hash = sha256(bundle_path)
    metadata = {
        "artifact": "Objective 6 private training-only PadChest-6 report labeler",
        "train_manifest_sha256": TRAIN_SHA256,
        "training_cases": len(train),
        "concepts": list(PAD_CHEST_6),
        "features": "Spanish character TF-IDF word-boundary n-grams 3-5",
        "models": "six independent class-balanced logistic regressions",
        "probability_threshold": 0.5,
        "validation_or_test_outputs_used_to_fit": False,
        "bundle_sha256": bundle_hash,
        "public_upload_allowed": False,
        "test_evaluated": False,
    }
    write_json(metadata, metadata_path)
    from huggingface_hub import CommitOperationAdd

    api.create_commit(
        repo_id=repository, repo_type="model", token=token,
        operations=[
            CommitOperationAdd(path_in_repo=remote_bundle, path_or_fileobj=str(bundle_path)),
            CommitOperationAdd(path_in_repo=remote_metadata, path_or_fileobj=str(metadata_path)),
            CommitOperationAdd(
                path_in_repo=remote_metadata + ".sha256",
                path_or_fileobj=str(metadata_path.with_suffix(".json.sha256")),
            ),
        ],
        commit_message="recovery: Objective 6 private training-only clinical labeler",
    )
    return vectorizer, models, bundle_hash, "fitted"


def label_reports(vectorizer: Any, models: list[Any], reports: list[str]) -> np.ndarray:
    features = vectorizer.transform([normalise_report(value) for value in reports])
    probabilities = np.column_stack([model.predict_proba(features)[:, 1] for model in models])
    return (probabilities >= 0.5).astype(np.int8)


def point_metrics(
    frame: pd.DataFrame,
    reference_tokens: list[list[str]],
    document_frequency: Any,
    train_report_set: set[str],
    reference_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    generated = frame["generated_report"].fillna("").astype(str).tolist()
    hypothesis_tokens = [tokenise_report(value) for value in generated]
    bleu = np.stack([
        bleu_statistics(reference, hypothesis)
        for reference, hypothesis in zip(reference_tokens, hypothesis_tokens)
    ])
    rouge = np.asarray([
        rouge_l_f1(reference, hypothesis)
        for reference, hypothesis in zip(reference_tokens, hypothesis_tokens)
    ])
    meteor = np.asarray([
        exact_token_meteor(reference, hypothesis)
        for reference, hypothesis in zip(reference_tokens, hypothesis_tokens)
    ])
    cider = np.asarray([
        cider_d_score(reference, hypothesis, document_frequency, len(reference_tokens))
        for reference, hypothesis in zip(reference_tokens, hypothesis_tokens)
    ])
    empty = np.asarray([not tokens for tokens in hypothesis_tokens], dtype=float)
    repeated = np.asarray([repeated_ngram(tokens) for tokens in hypothesis_tokens], dtype=float)
    exact = np.asarray([
        normalise_report(value) in train_report_set for value in generated
    ], dtype=float)
    contradictions = np.zeros(len(frame), dtype=float)
    mentions = np.zeros(len(frame), dtype=float)
    for index, (report, labels) in enumerate(zip(generated, reference_labels)):
        contradictions[index], mentions[index] = explicit_contradictions(report, labels)
    clinical = clinical_scores(reference_labels, predicted_labels)
    metrics = {
        **{f"BLEU-{order}": corpus_bleu(bleu, order) for order in range(1, 5)},
        "ROUGE-L": float(rouge.mean()),
        "METEOR_exact_token": float(meteor.mean()),
        "CIDEr-D": float(cider.mean()),
        **clinical,
        "explicit_negation_contradiction_rate": float(contradictions.sum() / mentions.sum()) if mentions.sum() else 0.0,
        "explicit_clinical_mentions": int(mentions.sum()),
        "empty_report_rate": float(empty.mean()),
        "repeated_4gram_report_rate": float(repeated.mean()),
        "training_report_exact_match_rate": float(exact.mean()),
        "unique_generated_report_fraction": float(len(set(map(normalise_report, generated))) / len(generated)),
    }
    arrays = {
        "bleu": bleu, "rouge": rouge, "meteor": meteor, "cider": cider,
        "empty": empty, "repeated": repeated, "exact": exact,
        "contradictions": contradictions, "mentions": mentions,
        "predicted_labels": predicted_labels,
    }
    return metrics, arrays


def metrics_for_indices(
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    reference_labels: np.ndarray,
) -> dict[str, float]:
    clinical = clinical_scores(reference_labels[indices], arrays["predicted_labels"][indices])
    contradiction_mentions = arrays["mentions"][indices].sum()
    return {
        **{f"BLEU-{order}": corpus_bleu(arrays["bleu"][indices], order) for order in range(1, 5)},
        "ROUGE-L": float(arrays["rouge"][indices].mean()),
        "METEOR_exact_token": float(arrays["meteor"][indices].mean()),
        "CIDEr-D": float(arrays["cider"][indices].mean()),
        "macro_concept_f1": clinical["macro_concept_f1"],
        "micro_concept_f1": clinical["micro_concept_f1"],
        "explicit_negation_contradiction_rate": float(arrays["contradictions"][indices].sum() / contradiction_mentions) if contradiction_mentions else 0.0,
        "empty_report_rate": float(arrays["empty"][indices].mean()),
        "repeated_4gram_report_rate": float(arrays["repeated"][indices].mean()),
        "training_report_exact_match_rate": float(arrays["exact"][indices].mean()),
    }


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    completed_lock = (
        args.output_dir / "public" / "FINAL_OBJECTIVE6_VALIDATION_COMPARISON.json"
    )
    if completed_lock.is_file():
        raise RuntimeError("Objective 6 validation comparison is already finalized")
    if args.bootstrap_replicates != 1000 or args.bootstrap_seed != 6042:
        raise RuntimeError("Objective 6 preregistered bootstrap configuration changed")
    protected = {
        args.train_manifest: TRAIN_SHA256,
        args.val_manifest: VAL_SHA256,
        args.validation_protocol: PROTOCOL_SHA256,
        args.validation_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 input changed: {path}")
    protocol = json.loads(args.validation_protocol.read_text(encoding="utf-8"))
    lock = json.loads(args.validation_lock.read_text(encoding="utf-8"))
    if (
        protocol.get("primary_system", {}).get("variant") != "multimodal"
        or lock.get("primary_system") != "multimodal"
        or lock.get("locked_test_evaluated") is not False
    ):
        raise RuntimeError("Objective 6 validation comparison lock changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.output_dir / "private"
    public_dir = args.output_dir / "public"
    private_dir.mkdir(exist_ok=True)
    public_dir.mkdir(exist_ok=True)

    train = pd.read_csv(args.train_manifest, low_memory=False).fillna("")
    validation = pd.read_csv(args.val_manifest, low_memory=False).fillna("")
    if len(train) != 29283 or len(validation) != 6280:
        raise RuntimeError("Objective 6 cohort size changed")
    roots = {
        "nearest_training_image_retrieval": args.retrieval_root,
        "image_only": args.image_only_root,
        "multimodal": args.multimodal_root,
    }
    frames = {name: load_system(root, name) for name, root in roots.items()}
    expected_case_codes = validation["case_code"].astype(str).tolist()
    for name, frame in frames.items():
        if frame["case_code"].astype(str).tolist() != expected_case_codes:
            raise RuntimeError(f"Objective 6 {name} order does not match validation manifest")
        if not (
            frame["reference_report"].map(normalise_report).to_numpy()
            == validation["report"].map(normalise_report).to_numpy()
        ).all():
            raise RuntimeError(f"Objective 6 {name} references changed")
        if not (
            frame["reference_labels"].astype(str).to_numpy()
            == validation["labels"].astype(str).to_numpy()
        ).all():
            raise RuntimeError(f"Objective 6 {name} reference labels changed")
        if not (
            frame["patient_id"].astype(str).to_numpy()
            == validation["patient_id"].astype(str).to_numpy()
        ).all():
            raise RuntimeError(f"Objective 6 {name} patient alignment changed")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    if not bool(api.model_info(args.private_hf_repo, token=token).private):
        raise RuntimeError("Objective 6 recovery repository must remain private")
    remote_root = args.private_hf_path.strip("/")
    print("\n--- FITTING OR RESTORING TRAINING-ONLY CLINICAL LABELER ---")
    vectorizer, models, labeler_hash, labeler_action = fit_or_restore_labeler(
        train, private_dir, api, args.private_hf_repo, remote_root, token
    )
    reference_labels = np.stack(validation["labels"].map(parse_padchest6_labels).to_numpy())
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
        print("Computing locked metrics:", name)
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

    patients = validation["patient_id"].astype(str).to_numpy()
    unique_patients = np.unique(patients)
    patient_rows = {patient: np.flatnonzero(patients == patient) for patient in unique_patients}
    random = np.random.default_rng(args.bootstrap_seed)
    metric_names = tuple(metrics_for_indices(np.arange(len(validation)), arrays[SYSTEMS[0]], reference_labels))
    distributions = {
        system: {metric: np.zeros(args.bootstrap_replicates) for metric in metric_names}
        for system in SYSTEMS
    }
    for replicate in range(args.bootstrap_replicates):
        sampled_patients = random.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([patient_rows[patient] for patient in sampled_patients])
        for system in SYSTEMS:
            values = metrics_for_indices(indices, arrays[system], reference_labels)
            for metric, value in values.items():
                distributions[system][metric][replicate] = value
        if (replicate + 1) % 100 == 0:
            print(f"Bootstrap replicate {replicate + 1}/{args.bootstrap_replicates}")

    intervals = {
        system: {metric: percentile_interval(values) for metric, values in metrics.items()}
        for system, metrics in distributions.items()
    }
    paired = {}
    for comparator in ("image_only", "nearest_training_image_retrieval"):
        paired[f"multimodal_minus_{comparator}"] = {}
        for metric in ("macro_concept_f1", "CIDEr-D"):
            difference = distributions["multimodal"][metric] - distributions[comparator][metric]
            paired[f"multimodal_minus_{comparator}"][metric] = {
                "point_difference": float(point["multimodal"][metric] - point[comparator][metric]),
                "bootstrap_95_ci": percentile_interval(difference),
                "two_sided_bootstrap_p": float(min(1.0, 2.0 * min(
                    np.mean(difference <= 0), np.mean(difference >= 0)
                ))),
            }

    case_metrics = pd.concat(private_frames, ignore_index=True)
    case_path = private_dir / "validation_case_metrics_private.csv"
    case_metrics.to_csv(case_path, index=False, lineterminator="\n")
    case_hash = write_checksum(case_path)
    bootstrap_path = private_dir / "validation_bootstrap_private.npz"
    np.savez_compressed(
        bootstrap_path,
        **{
            f"{system}__{metric}": values
            for system, metrics in distributions.items() for metric, values in metrics.items()
        },
    )
    bootstrap_hash = write_checksum(bootstrap_path)
    summary = {
        "artifact": "Objective 6 locked validation report-generation comparison",
        "version": "v1.0.0",
        "validation_cases": len(validation),
        "validation_patients": int(validation["patient_id"].astype(str).nunique()),
        "systems": point,
        "patient_cluster_bootstrap_95_ci": intervals,
        "paired_primary_comparisons": paired,
        "clinical_labeler": {
            "training_cases": len(train),
            "training_manifest_sha256": TRAIN_SHA256,
            "bundle_sha256": labeler_hash,
            "recovery_action": labeler_action,
            "validation_reference_report_performance": labeler_validation,
            "threshold": 0.5,
            "validation_or_test_outputs_used_to_fit": False,
        },
        "primary_system": "multimodal",
        "primary_system_changed_after_validation": False,
        "primary_metric": "macro_concept_f1",
        "secondary_metric": "CIDEr-D",
        "meteor_variant": "Spanish exact-token METEOR; no English stemming or synonym resources",
        "bleu_smoothing": "deterministic add-one smoothing",
        "bootstrap": {
            "method": "paired patient-cluster percentile bootstrap",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
        "privacy": {
            "raw_reports_public": False,
            "generated_reports_public": False,
            "case_level_metrics_public": False,
            "patient_or_image_identifiers_public": False,
        },
        "private_artifact_sha256": {
            "case_metrics": case_hash,
            "bootstrap": bootstrap_hash,
        },
        "locked_test_manifest_opened": False,
        "locked_test_reports_accessed": False,
        "locked_test_evaluated": False,
    }
    summary_path = public_dir / "objective6_validation_comparison_summary_public.json"
    summary_hash = write_json(summary, summary_path)

    labels = ["Retrieval", "Image-only", "Multimodal"]
    colors = ["#8c8c8c", "#3b82f6", "#0f9d76"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, metric, title in zip(
        axes, ("macro_concept_f1", "CIDEr-D"),
        ("PadChest-6 macro concept F1", "CIDEr-D"),
    ):
        values = [point[system][metric] for system in SYSTEMS]
        lower = [values[index] - intervals[system][metric][0] for index, system in enumerate(SYSTEMS)]
        upper = [intervals[system][metric][1] - values[index] for index, system in enumerate(SYSTEMS)]
        axis.bar(labels, values, color=colors, yerr=np.asarray([lower, upper]), capsize=4)
        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Objective 6: Locked Validation Report-Generation Comparison")
    figure.tight_layout()
    figure_path = public_dir / "objective6_validation_comparison.png"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    figure_hash = sha256(figure_path)
    lock_payload = {
        "artifact": "Final Objective 6 validation comparison lock",
        "immutable": True,
        "validation_protocol_sha256": PROTOCOL_SHA256,
        "validation_lock_sha256": LOCK_SHA256,
        "summary_sha256": summary_hash,
        "figure_sha256": figure_hash,
        "primary_system": "multimodal",
        "primary_system_changed": False,
        "validation_evaluation_count": 1,
        "locked_test_evaluated": False,
        "locked_test_evaluation_count": 0,
    }
    final_lock_path = public_dir / "FINAL_OBJECTIVE6_VALIDATION_COMPARISON.json"
    final_lock_hash = write_json(lock_payload, final_lock_path)

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
        commit_message="recovery: Objective 6 private validation comparison",
    )
    print("\n--- OBJECTIVE 6 LOCKED VALIDATION RESULTS ---")
    for system in SYSTEMS:
        print("SYSTEM:", system)
        print("Macro concept F1:", point[system]["macro_concept_f1"])
        print("CIDEr-D:", point[system]["CIDEr-D"])
        print("BLEU-4:", point[system]["BLEU-4"])
        print("ROUGE-L:", point[system]["ROUGE-L"])
        print("METEOR exact-token:", point[system]["METEOR_exact_token"])
    print("\n--- FINAL VERIFICATION ---")
    print("Summary SHA-256:", summary_hash)
    print("Figure SHA-256:", figure_hash)
    print("Final-lock SHA-256:", final_lock_hash)
    print("Clinical labeler action:", labeler_action)
    print("Private recovery verified:", True)
    print("Raw reports printed:", False)
    print("Case-level outputs publicly uploaded:", False)
    print("Locked-test manifest opened:", False)
    print("Locked-test reports accessed:", False)
    print("Locked-test evaluated:", False)
    print("OBJECTIVE 6 LOCKED VALIDATION COMPARISON SUCCESSFUL")


if __name__ == "__main__":
    main()
