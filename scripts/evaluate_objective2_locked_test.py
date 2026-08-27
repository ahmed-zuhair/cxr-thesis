#!/usr/bin/env python3
"""Evaluate five frozen Objective 2 candidates once on the locked test cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective2.data import (
    GraphClassificationDataset,
    ImageClassificationDataset,
    collate_graph_samples,
)
from cxr_thesis.objective2.evaluation import paired_bootstrap_comparison
from cxr_thesis.objective2.metrics import multilabel_metrics
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective2.training import predict, seed_everything


PRIMARY_LABELS = [
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Consolidation",
    "Pneumothorax",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Emphysema",
    "Edema",
    "Fibrosis",
]
MODEL_ORDER = ("cnn", "attention_cnn", "vit", "gcn", "gat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    for model in MODEL_ORDER:
        option = model.replace("_", "-")
        parser.add_argument(f"--{option}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--expected-{option}-sha256", required=True)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--expected-test-cases", type=int, default=5_000)
    parser.add_argument("--expected-test-patients", type=int, default=541)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--graph-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--reference-model", choices=MODEL_ORDER, default="cnn")
    parser.add_argument("--private-hf-repo")
    parser.add_argument("--private-hf-path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def candidate_arguments(args: argparse.Namespace, model: str) -> tuple[Path, str]:
    key = model.replace("_", "-")
    checkpoint = getattr(args, f"{model}_checkpoint")
    expected_hash = getattr(args, f"expected_{model}_sha256")
    if key != model:
        # argparse converts hyphens to underscores, so the attributes above remain valid.
        pass
    return checkpoint, expected_hash


def validate_candidate(
    model: str,
    checkpoint_path: Path,
    expected_hash: str,
) -> tuple[dict[str, object], np.ndarray]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"{model} checkpoint SHA-256 does not match")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != model:
        raise RuntimeError(f"{model} checkpoint has the wrong model identity")
    if checkpoint.get("test_evaluated") is not False:
        raise RuntimeError(f"{model} candidate is not test-blind")
    if checkpoint.get("label_names") != PRIMARY_LABELS:
        raise RuntimeError(f"{model} checkpoint label order does not match")
    thresholds = np.asarray(checkpoint.get("validation_thresholds"), dtype=np.float64)
    if thresholds.shape != (len(PRIMARY_LABELS),):
        raise RuntimeError(f"{model} has invalid validation thresholds")
    if not np.isfinite(thresholds).all() or np.any((thresholds <= 0) | (thresholds >= 1)):
        raise RuntimeError(f"{model} validation thresholds are not finite probabilities")
    if "model_state" not in checkpoint:
        raise RuntimeError(f"{model} checkpoint is missing model_state")
    return checkpoint, thresholds


def make_loader(
    model: str,
    frame: pd.DataFrame,
    label_columns: list[str],
    args: argparse.Namespace,
) -> DataLoader:
    if model in {"gcn", "gat"}:
        dataset = GraphClassificationDataset(frame, label_columns, args.graph_root)
        batch_size = args.graph_batch_size
        collate = collate_graph_samples
    else:
        dataset = ImageClassificationDataset(
            frame,
            label_columns,
            data_root=args.data_root,
            image_size=args.image_size,
            augment=False,
            seed=args.seed,
        )
        batch_size = args.image_batch_size
        collate = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        persistent_workers=args.workers > 0,
    )


def make_comparison_figure(summary: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    models = list(MODEL_ORDER)
    display = [model.replace("_", "-").upper() for model in models]
    metric_names = ("auroc", "auprc", "f1")
    titles = ("Macro AUROC", "Macro AUPRC", "Macro F1")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    intervals = summary["bootstrap"]["model_metric_95_ci"]
    for axis, metric, title in zip(axes, metric_names, titles):
        values = [summary["models"][model]["test_metrics"]["macro"][metric] for model in models]
        low = [intervals[model][metric][0] for model in models]
        high = [intervals[model][metric][1] for model in models]
        errors = np.asarray(
            [[value - lower for value, lower in zip(values, low)],
             [upper - value for value, upper in zip(values, high)]],
            dtype=float,
        )
        axis.bar(display, values, color=["#3569a8", "#5c8bc3", "#7b68ae", "#4c9a70", "#c05a47"])
        axis.errorbar(display, values, yerr=errors, fmt="none", ecolor="black", capsize=4)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Objective 2 Locked-Test Comparison (Frozen Validation Candidates)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "cxr-thesis"})
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    if args.image_batch_size <= 0 or args.graph_batch_size <= 0 or args.workers < 0:
        raise ValueError("Batch sizes must be positive and workers nonnegative")
    final_lock = args.output_dir / "FINAL_LOCKED_TEST_EVALUATION.json"
    if final_lock.is_file():
        raise RuntimeError("The locked-test evaluation is already finalized")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Output exists; use --resume only for an interrupted run: {args.output_dir}"
        )
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError("Resume output directory does not exist")

    # All candidate identities, hashes and validation thresholds are frozen before
    # the test manifest is loaded or any test label value is accessed.
    checkpoints: dict[str, dict[str, object]] = {}
    thresholds: dict[str, np.ndarray] = {}
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_paths: dict[str, Path] = {}
    for model in MODEL_ORDER:
        checkpoint_path, expected_hash = candidate_arguments(args, model)
        checkpoint, model_thresholds = validate_candidate(
            model, checkpoint_path, expected_hash
        )
        checkpoints[model] = checkpoint
        thresholds[model] = model_thresholds
        checkpoint_hashes[model] = expected_hash
        checkpoint_paths[model] = checkpoint_path

    test_hash = sha256_file(args.test_manifest)
    if test_hash != args.expected_test_sha256:
        raise RuntimeError("Locked-test manifest SHA-256 does not match")
    frame = pd.read_csv(
        args.test_manifest,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    if len(frame) != args.expected_test_cases:
        raise RuntimeError("Locked-test case count does not match")
    if set(frame["split"].astype(str).str.lower()) != {"test"}:
        raise RuntimeError("Locked-test manifest contains a non-test split")
    if int(frame["patient_id"].nunique()) != args.expected_test_patients:
        raise RuntimeError("Locked-test patient count does not match")
    if frame["image_id"].astype(str).duplicated().any():
        raise RuntimeError("Locked-test manifest contains duplicate image IDs")
    label_columns = [f"label_{label}" for label in PRIMARY_LABELS]
    missing_labels = sorted(set(label_columns) - set(frame.columns))
    if missing_labels:
        raise RuntimeError(f"Locked-test labels are missing: {missing_labels}")
    target_array = frame[label_columns].to_numpy(dtype=np.int8)
    if not np.isin(target_array, [0, 1]).all():
        raise RuntimeError("Locked-test targets are not binary")
    case_order_hash = sha256_text("\n".join(frame["image_id"].astype(str)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_root = args.output_dir / "private"
    public_root = args.output_dir / "public"
    private_root.mkdir(exist_ok=True)
    public_root.mkdir(exist_ok=True)
    state_path = private_root / "evaluation_state_private.json"
    signature = {
        "format_version": 1,
        "test_manifest_sha256": test_hash,
        "test_cases": len(frame),
        "test_patients": int(frame["patient_id"].nunique()),
        "case_order_sha256": case_order_hash,
        "checkpoint_sha256": checkpoint_hashes,
        "reference_model": args.reference_model,
        "seed": args.seed,
        "image_size": args.image_size,
    }
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise RuntimeError("Interrupted evaluation signature does not match")
    else:
        state = {"signature": signature, "completed_models": [], "inference": {}}
        atomic_json(state, state_path)

    if bool(args.private_hf_repo) != bool(args.private_hf_path):
        raise ValueError("--private-hf-repo and --private-hf-path must be supplied together")
    private_api = None
    private_token = ""
    private_remote_files: set[str] = set()
    private_cache = private_root / "hf_downloads"
    if args.private_hf_repo:
        private_token = os.environ.get("HF_TOKEN", "").strip()
        if not private_token:
            raise RuntimeError("HF_TOKEN is required for private evaluation recovery")
        from huggingface_hub import HfApi

        private_api = HfApi(token=private_token)
        private_info = private_api.model_info(args.private_hf_repo, token=private_token)
        if not bool(private_info.private):
            raise RuntimeError("Locked-test evaluation recovery repository must be private")
        private_remote_files = set(
            private_api.list_repo_files(
                args.private_hf_repo, repo_type="model", token=private_token
            )
        )
        private_cache.mkdir(parents=True, exist_ok=True)
        remote_state = f"{args.private_hf_path.strip('/')}/evaluation_state_private.json"
        if remote_state in private_remote_files:
            from huggingface_hub import hf_hub_download

            downloaded_state = Path(
                hf_hub_download(
                    args.private_hf_repo,
                    filename=remote_state,
                    repo_type="model",
                    token=private_token,
                    local_dir=private_cache,
                    force_download=True,
                )
            )
            recovered_state = json.loads(downloaded_state.read_text(encoding="utf-8"))
            if recovered_state.get("signature") != signature:
                raise RuntimeError("Private evaluation recovery signature does not match")
            state = recovered_state
            atomic_json(state, state_path)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities: dict[str, np.ndarray] = {}
    completed_models = set(state.get("completed_models", []))
    print(
        json.dumps(
            {
                "event": "locked_test_evaluation_start",
                "models": list(MODEL_ORDER),
                "completed_models_restored": sorted(completed_models),
                "test_cases": len(frame),
                "test_patients": int(frame["patient_id"].nunique()),
                "validation_thresholds_frozen": True,
                "checkpoint_hashes_frozen": True,
                "test_threshold_tuning": False,
                "device": str(device),
            },
            indent=2,
        )
    )

    for model_name in MODEL_ORDER:
        prediction_path = private_root / f"{model_name}_predictions_private.npz"
        prediction_checksum = prediction_path.with_suffix(".npz.sha256")
        remote_prediction = (
            f"{args.private_hf_path.strip('/')}/models/{prediction_path.name}"
            if args.private_hf_path
            else ""
        )
        remote_checksum = f"{remote_prediction}.sha256" if remote_prediction else ""
        if private_api is not None and (
            (remote_prediction in private_remote_files)
            != (remote_checksum in private_remote_files)
        ):
            raise RuntimeError(
                f"Incomplete private locked-test recovery pair for {model_name}"
            )
        if (
            not prediction_path.is_file()
            and private_api is not None
            and remote_prediction in private_remote_files
            and remote_checksum in private_remote_files
        ):
            from huggingface_hub import hf_hub_download

            downloaded_checksum = Path(
                hf_hub_download(
                    args.private_hf_repo,
                    filename=remote_checksum,
                    repo_type="model",
                    token=private_token,
                    local_dir=private_cache,
                    force_download=True,
                )
            )
            expected_prediction_hash = downloaded_checksum.read_text(
                encoding="utf-8"
            ).split()[0]
            downloaded_prediction = Path(
                hf_hub_download(
                    args.private_hf_repo,
                    filename=remote_prediction,
                    repo_type="model",
                    token=private_token,
                    local_dir=private_cache,
                    force_download=True,
                )
            )
            if sha256_file(downloaded_prediction) != expected_prediction_hash:
                raise RuntimeError(f"Private {model_name} recovery hash does not match")
            prediction_path.write_bytes(downloaded_prediction.read_bytes())
            prediction_checksum.write_text(
                f"{expected_prediction_hash}  {prediction_path.name}\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {"event": "model_predictions_private_hf_restored", "model": model_name}
                )
            )
        if prediction_path.is_file():
            saved = np.load(prediction_path)
            model_probabilities = np.asarray(saved["probabilities"], dtype=np.float32)
            saved_targets = np.asarray(saved["targets"], dtype=np.int8)
            if model_probabilities.shape != target_array.shape or not np.array_equal(
                saved_targets, target_array
            ):
                raise RuntimeError(f"Interrupted {model_name} predictions are incompatible")
            probabilities[model_name] = model_probabilities
            completed_models.add(model_name)
            print(json.dumps({"event": "model_predictions_restored", "model": model_name}))
            continue

        model = build_classifier(
            model_name,
            len(PRIMARY_LABELS),
            image_size=args.image_size,
            node_dim=7,
            clinical_dim=9,
        ).to(device)
        model.load_state_dict(checkpoints[model_name]["model_state"], strict=True)
        loader = make_loader(model_name, frame, label_columns, args)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model_probabilities, observed_targets = predict(model, loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if not np.array_equal(observed_targets.astype(np.int8), target_array):
            raise RuntimeError(f"{model_name} target order changed during evaluation")
        atomic_npz(
            prediction_path,
            probabilities=model_probabilities.astype(np.float32),
            targets=target_array.astype(np.int8),
        )
        prediction_hash = sha256_file(prediction_path)
        prediction_checksum.write_text(
            f"{prediction_hash}  {prediction_path.name}\n", encoding="utf-8"
        )
        probabilities[model_name] = model_probabilities.astype(np.float32)
        completed_models.add(model_name)
        state["completed_models"] = [
            model for model in MODEL_ORDER if model in completed_models
        ]
        state["inference"][model_name] = {
            "seconds": elapsed,
            "cases_per_second": len(frame) / elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        }
        atomic_json(state, state_path)
        if private_api is not None:
            from huggingface_hub import CommitOperationAdd

            remote_state = (
                f"{args.private_hf_path.strip('/')}/evaluation_state_private.json"
            )
            private_api.create_commit(
                repo_id=args.private_hf_repo,
                repo_type="model",
                token=private_token,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=remote_prediction,
                        path_or_fileobj=str(prediction_path),
                    ),
                    CommitOperationAdd(
                        path_in_repo=remote_checksum,
                        path_or_fileobj=str(prediction_checksum),
                    ),
                    CommitOperationAdd(
                        path_in_repo=remote_state,
                        path_or_fileobj=str(state_path),
                    ),
                ],
                commit_message=f"recovery: save locked-test {model_name} predictions",
            )
            private_remote_files.update(
                {remote_prediction, remote_checksum, remote_state}
            )
        print(
            json.dumps(
                {
                    "event": "model_locked_test_complete",
                    "model": model_name,
                    **state["inference"][model_name],
                    "test_threshold_tuning": False,
                }
            )
        )
        del model, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if set(probabilities) != set(MODEL_ORDER):
        raise RuntimeError("Not all frozen candidates produced locked-test predictions")
    model_results: dict[str, object] = {}
    for model_name in MODEL_ORDER:
        metrics = multilabel_metrics(
            probabilities[model_name], target_array, thresholds=thresholds[model_name]
        )
        for label, label_metrics in zip(PRIMARY_LABELS, metrics["per_label"]):
            label_metrics["label"] = label
        model_results[model_name] = {
            "checkpoint_sha256": checkpoint_hashes[model_name],
            "best_validation_epoch": int(checkpoints[model_name]["epoch"]),
            "validation_macro_auroc": float(
                checkpoints[model_name]["validation_metrics"]["macro"]["auroc"]
            ),
            "validation_thresholds": thresholds[model_name].tolist(),
            "test_metrics": metrics,
            "inference": state["inference"].get(model_name, {}),
        }

    bootstrap = paired_bootstrap_comparison(
        probabilities,
        target_array,
        thresholds,
        reference_model=args.reference_model,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    summary = {
        "artifact": "Objective 2 single locked-test comparison",
        "models": model_results,
        "model_order_frozen_before_test": list(MODEL_ORDER),
        "reference_model_frozen_before_test": args.reference_model,
        "reference_selection_basis": "highest validation macro AUROC",
        "test_manifest_sha256": test_hash,
        "test_cases": len(frame),
        "test_patients": int(frame["patient_id"].nunique()),
        "labels": PRIMARY_LABELS,
        "aggregate_positive_counts": {
            label: int(target_array[:, index].sum())
            for index, label in enumerate(PRIMARY_LABELS)
        },
        "bootstrap": bootstrap,
        "validation_thresholds_reused_without_change": True,
        "test_threshold_tuning": False,
        "test_used_for_model_selection": False,
        "training_repeated": False,
        "test_evaluated": True,
        "test_evaluation_count_per_model": 1,
        "case_level_predictions_publication_allowed": False,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "medical_images_included": False,
        "private_manifests_included": False,
        "private_hf_recovery_enabled": private_api is not None,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    summary_path = public_root / "objective2_locked_test_summary_public.json"
    atomic_json(summary, summary_path)
    figure_path = public_root / "objective2_locked_test_model_comparison.png"
    make_comparison_figure(summary, figure_path)
    lock_payload = {
        "artifact": "Objective 2 immutable locked-test evaluation record",
        "test_manifest_sha256": test_hash,
        "checkpoint_sha256": checkpoint_hashes,
        "summary_sha256": sha256_file(summary_path),
        "figure_sha256": sha256_file(figure_path),
        "completed_models": list(MODEL_ORDER),
        "validation_thresholds_reused_without_change": True,
        "test_used_for_model_selection": False,
        "test_evaluated": True,
    }
    atomic_json(lock_payload, final_lock)
    if private_api is not None:
        from huggingface_hub import CommitOperationAdd

        private_api.create_commit(
            repo_id=args.private_hf_repo,
            repo_type="model",
            token=private_token,
            operations=[
                CommitOperationAdd(
                    path_in_repo=(
                        f"{args.private_hf_path.strip('/')}/FINAL_LOCKED_TEST_EVALUATION.json"
                    ),
                    path_or_fileobj=str(final_lock),
                ),
                CommitOperationAdd(
                    path_in_repo=(
                        f"{args.private_hf_path.strip('/')}/{summary_path.name}"
                    ),
                    path_or_fileobj=str(summary_path),
                ),
                CommitOperationAdd(
                    path_in_repo=(
                        f"{args.private_hf_path.strip('/')}/{figure_path.name}"
                    ),
                    path_or_fileobj=str(figure_path),
                ),
            ],
            commit_message="recovery: finalize Objective 2 locked-test evaluation",
        )
    print(json.dumps(json_safe(lock_payload), indent=2, sort_keys=True))
    print("OBJECTIVE 2 SINGLE LOCKED-TEST EVALUATION SUCCESSFUL")


if __name__ == "__main__":
    main()
