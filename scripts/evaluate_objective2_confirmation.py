#!/usr/bin/env python3
"""Evaluate the frozen CNN and DenseNet once on the independent cohort."""

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

from cxr_thesis.objective2.data import ImageClassificationDataset
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
MODEL_ORDER = ("cnn", "densenet121")
FINAL_LOCK_NAME = "FINAL_INDEPENDENT_CONFIRMATION_EVALUATION.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--protocol-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--cnn-checkpoint", type=Path, required=True)
    parser.add_argument("--densenet121-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-cnn-sha256", required=True)
    parser.add_argument("--expected-densenet121-sha256", required=True)
    parser.add_argument("--expected-confirmation-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-confirmation-cases", type=int, default=5_000)
    parser.add_argument("--expected-confirmation-patients", type=int, default=568)
    parser.add_argument("--cnn-batch-size", type=int, default=64)
    parser.add_argument("--densenet121-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--private-hf-repo", required=True)
    parser.add_argument("--private-hf-path", required=True)
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


def validate_protocol(
    payload: dict[str, object],
    *,
    confirmation_hash: str,
    cases: int,
    patients: int,
) -> None:
    checks = {
        "manifest_hash": payload.get("confirmation_manifest_sha256")
        == confirmation_hash,
        "cases": payload.get("confirmation_cases") == cases,
        "patients": payload.get("confirmation_patients") == patients,
        "zero_overlap": payload.get("patient_overlap_with_original_locked_test") == 0,
        "label_blind": payload.get("selection_used_labels") is False,
        "prediction_blind": payload.get("selection_used_predictions") is False,
        "risk_blind": payload.get("selection_used_risk_scores") is False,
        "pre_evaluation_lock": payload.get("status")
        == "locked before confirmation-label evaluation",
        "post_test_enhancement_disclosed": payload.get(
            "enhancement_developed_after_original_locked_test"
        )
        is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Independent confirmation protocol is invalid: {checks}")


def validate_final_lock_payload(
    payload: dict[str, object],
    *,
    confirmation_hash: str,
    protocol_hash: str,
    checkpoint_hashes: dict[str, str],
) -> None:
    checks = {
        "confirmation_hash": payload.get("confirmation_manifest_sha256")
        == confirmation_hash,
        "protocol_hash": payload.get("protocol_sha256") == protocol_hash,
        "checkpoint_hashes": payload.get("checkpoint_sha256") == checkpoint_hashes,
        "completed_models": payload.get("completed_models") == list(MODEL_ORDER),
        "thresholds_frozen": payload.get("validation_thresholds_reused_without_change")
        is True,
        "confirmation_not_selection": payload.get(
            "confirmation_used_for_model_selection"
        )
        is False,
        "evaluated": payload.get("confirmation_evaluated") is True,
        "evaluation_count": payload.get("confirmation_evaluation_count") == 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Private confirmation final lock is invalid: {checks}")


def validate_candidate(
    model_name: str, checkpoint_path: Path, expected_hash: str
) -> tuple[dict[str, object], np.ndarray]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_file(checkpoint_path) != expected_hash:
        raise RuntimeError(f"{model_name} checkpoint SHA-256 does not match")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != model_name:
        raise RuntimeError(f"{model_name} checkpoint identity does not match")
    if checkpoint.get("test_evaluated") is not False:
        raise RuntimeError(f"{model_name} checkpoint is not test-blind")
    if checkpoint.get("label_names") != PRIMARY_LABELS:
        raise RuntimeError(f"{model_name} checkpoint label order does not match")
    thresholds = np.asarray(checkpoint.get("validation_thresholds"), dtype=np.float64)
    if thresholds.shape != (len(PRIMARY_LABELS),) or not np.isfinite(thresholds).all():
        raise RuntimeError(f"{model_name} validation thresholds are invalid")
    if np.any((thresholds <= 0.0) | (thresholds >= 1.0)):
        raise RuntimeError(f"{model_name} validation thresholds are not probabilities")
    if "model_state" not in checkpoint:
        raise RuntimeError(f"{model_name} checkpoint has no model state")
    return checkpoint, thresholds


def model_input_config(
    model_name: str, checkpoint: dict[str, object]
) -> dict[str, object]:
    configuration = dict(checkpoint.get("model_config") or {})
    signature = dict(checkpoint.get("training_signature") or {})
    image_size = int(configuration.get("image_size", signature.get("image_size", 224)))
    dropout = float(configuration.get("dropout", signature.get("dropout", 0.2)))
    if model_name == "densenet121":
        expected = {"channels": 3, "normalisation": "imagenet"}
        if image_size != 320:
            raise RuntimeError("DenseNet confirmation input size must remain 320")
    else:
        expected = {"channels": 1, "normalisation": "unit"}
        if image_size != 224:
            raise RuntimeError("CNN confirmation input size must remain 224")
    return {"image_size": image_size, "dropout": dropout, **expected}


def make_loader(
    frame: pd.DataFrame,
    label_columns: list[str],
    args: argparse.Namespace,
    input_config: dict[str, object],
    model_name: str,
) -> DataLoader:
    dataset = ImageClassificationDataset(
        frame,
        label_columns,
        data_root=args.data_root,
        image_size=int(input_config["image_size"]),
        augment=False,
        seed=args.seed,
        output_channels=int(input_config["channels"]),
        normalisation=str(input_config["normalisation"]),
    )
    batch_size = (
        args.densenet121_batch_size
        if model_name == "densenet121"
        else args.cnn_batch_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )


def make_figure(summary: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ("auroc", "auprc", "f1")
    titles = ("Macro AUROC", "Macro AUPRC", "Macro F1")
    intervals = summary["bootstrap"]["model_metric_95_ci"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for axis, metric, title in zip(axes, metrics, titles):
        values = [
            summary["models"][model]["confirmation_metrics"]["macro"][metric]
            for model in MODEL_ORDER
        ]
        low = [intervals[model][metric][0] for model in MODEL_ORDER]
        high = [intervals[model][metric][1] for model in MODEL_ORDER]
        errors = np.asarray(
            [
                [value - lower for value, lower in zip(values, low)],
                [upper - value for value, upper in zip(values, high)],
            ]
        )
        axis.bar(
            ["Original CNN", "Enhanced DenseNet-121"],
            values,
            color=["#3569a8", "#d67835"],
        )
        axis.errorbar(
            range(2),
            values,
            yerr=errors,
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Objective 2 Independent Confirmation (Frozen Models)")
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "cxr-thesis"},
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if min(args.cnn_batch_size, args.densenet121_batch_size) <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.workers < 0 or args.bootstrap_replicates <= 0:
        raise ValueError(
            "Workers must be nonnegative and bootstrap replicates positive"
        )

    final_lock = args.output_dir / FINAL_LOCK_NAME
    if final_lock.is_file():
        raise RuntimeError("Independent confirmation is already finalized locally")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume only after interruption")
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError("Resume output directory does not exist")

    protocol_hash = sha256_file(args.protocol_record)
    if protocol_hash != args.expected_protocol_sha256:
        raise RuntimeError("Published protocol SHA-256 does not match")
    protocol = json.loads(args.protocol_record.read_text(encoding="utf-8"))
    validate_protocol(
        protocol,
        confirmation_hash=args.expected_confirmation_sha256,
        cases=args.expected_confirmation_cases,
        patients=args.expected_confirmation_patients,
    )

    checkpoint_paths = {
        "cnn": args.cnn_checkpoint,
        "densenet121": args.densenet121_checkpoint,
    }
    checkpoint_hashes = {
        "cnn": args.expected_cnn_sha256,
        "densenet121": args.expected_densenet121_sha256,
    }
    checkpoints: dict[str, dict[str, object]] = {}
    thresholds: dict[str, np.ndarray] = {}
    input_configs: dict[str, dict[str, object]] = {}
    for model_name in MODEL_ORDER:
        checkpoint, frozen_thresholds = validate_candidate(
            model_name, checkpoint_paths[model_name], checkpoint_hashes[model_name]
        )
        checkpoints[model_name] = checkpoint
        thresholds[model_name] = frozen_thresholds
        input_configs[model_name] = model_input_config(model_name, checkpoint)

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for private confirmation recovery")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    if not bool(api.model_info(args.private_hf_repo, token=token).private):
        raise RuntimeError("Confirmation recovery repository must remain private")
    remote_root = args.private_hf_path.strip("/")
    remote_files = set(
        api.list_repo_files(args.private_hf_repo, repo_type="model", token=token)
    )
    remote_lock = f"{remote_root}/{FINAL_LOCK_NAME}"
    if remote_lock in remote_files:
        downloaded = Path(
            hf_hub_download(
                args.private_hf_repo,
                filename=remote_lock,
                repo_type="model",
                token=token,
                force_download=True,
            )
        )
        validate_final_lock_payload(
            json.loads(downloaded.read_text(encoding="utf-8")),
            confirmation_hash=args.expected_confirmation_sha256,
            protocol_hash=protocol_hash,
            checkpoint_hashes=checkpoint_hashes,
        )
        raise RuntimeError(
            "Independent confirmation is already finalized in private recovery; "
            "a second evaluation is forbidden"
        )

    # The private cohort is opened only after protocol, candidates, hashes and the
    # remote immutable-lock guard have all passed.
    confirmation_hash = sha256_file(args.confirmation_manifest)
    if confirmation_hash != args.expected_confirmation_sha256:
        raise RuntimeError("Confirmation manifest SHA-256 does not match")
    frame = pd.read_csv(
        args.confirmation_manifest,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    if len(frame) != args.expected_confirmation_cases:
        raise RuntimeError("Confirmation case count does not match")
    if int(frame["patient_id"].nunique()) != args.expected_confirmation_patients:
        raise RuntimeError("Confirmation patient count does not match")
    if set(frame["split"].astype(str).str.lower()) != {"test"}:
        raise RuntimeError("Confirmation cohort contains a non-test split")
    if frame["image_id"].astype(str).duplicated().any():
        raise RuntimeError("Confirmation cohort contains duplicate images")
    label_columns = [f"label_{label}" for label in PRIMARY_LABELS]
    missing = sorted(set(label_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Confirmation labels are missing: {missing}")
    targets = frame[label_columns].to_numpy(dtype=np.int8)
    if not np.isin(targets, [0, 1]).all():
        raise RuntimeError("Confirmation labels are not binary")
    case_order_hash = sha256_text("\n".join(frame["image_id"].astype(str)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_root = args.output_dir / "private"
    public_root = args.output_dir / "public"
    private_root.mkdir(exist_ok=True)
    public_root.mkdir(exist_ok=True)
    signature = {
        "format_version": 1,
        "confirmation_manifest_sha256": confirmation_hash,
        "protocol_sha256": protocol_hash,
        "checkpoint_sha256": checkpoint_hashes,
        "case_order_sha256": case_order_hash,
        "models": list(MODEL_ORDER),
        "seed": args.seed,
    }
    state_path = private_root / "confirmation_evaluation_state_private.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise RuntimeError(
                "Interrupted confirmation state signature does not match"
            )
    else:
        state = {"signature": signature, "completed_models": [], "inference": {}}
        atomic_json(state, state_path)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities: dict[str, np.ndarray] = {}
    recovery_cache = private_root / "hf_recovery_cache"
    for model_name in MODEL_ORDER:
        prediction_path = private_root / f"{model_name}_confirmation_predictions.npz"
        checksum_path = prediction_path.with_suffix(".npz.sha256")
        remote_prediction = f"{remote_root}/{prediction_path.name}"
        remote_checksum = f"{remote_root}/{checksum_path.name}"
        if not prediction_path.is_file() and remote_prediction in remote_files:
            if remote_checksum not in remote_files:
                raise RuntimeError(f"Incomplete private recovery for {model_name}")
            downloaded_checksum = Path(
                hf_hub_download(
                    args.private_hf_repo,
                    filename=remote_checksum,
                    repo_type="model",
                    token=token,
                    local_dir=recovery_cache,
                    force_download=True,
                )
            )
            expected = downloaded_checksum.read_text(encoding="utf-8").split()[0]
            downloaded_prediction = Path(
                hf_hub_download(
                    args.private_hf_repo,
                    filename=remote_prediction,
                    repo_type="model",
                    token=token,
                    local_dir=recovery_cache,
                    force_download=True,
                )
            )
            if sha256_file(downloaded_prediction) != expected:
                raise RuntimeError(f"Private {model_name} recovery hash does not match")
            prediction_path.write_bytes(downloaded_prediction.read_bytes())
            checksum_path.write_text(
                f"{expected}  {prediction_path.name}\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {"event": "private_predictions_restored", "model": model_name}
                )
            )

        if prediction_path.is_file():
            saved = np.load(prediction_path)
            restored_probabilities = np.asarray(
                saved["probabilities"], dtype=np.float32
            )
            restored_targets = np.asarray(saved["targets"], dtype=np.int8)
            restored_order = str(saved["case_order_sha256"].item())
            if (
                restored_probabilities.shape != targets.shape
                or not np.array_equal(restored_targets, targets)
                or restored_order != case_order_hash
            ):
                raise RuntimeError(
                    f"Interrupted {model_name} predictions are incompatible"
                )
            probabilities[model_name] = restored_probabilities
            if model_name not in state["completed_models"]:
                state["completed_models"].append(model_name)
                atomic_json(state, state_path)
            continue

        config = input_configs[model_name]
        model = build_classifier(
            model_name,
            len(PRIMARY_LABELS),
            image_size=int(config["image_size"]),
            clinical_dim=9,
            pretrained=False,
            dropout=float(config["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoints[model_name]["model_state"], strict=True)
        loader = make_loader(frame, label_columns, args, config, model_name)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model_probabilities, observed_targets = predict(model, loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if not np.array_equal(observed_targets.astype(np.int8), targets):
            raise RuntimeError(f"{model_name} changed confirmation case order")
        atomic_npz(
            prediction_path,
            probabilities=model_probabilities.astype(np.float32),
            targets=targets,
            case_order_sha256=np.asarray(case_order_hash),
        )
        prediction_hash = sha256_file(prediction_path)
        checksum_path.write_text(
            f"{prediction_hash}  {prediction_path.name}\n", encoding="utf-8"
        )
        probabilities[model_name] = model_probabilities.astype(np.float32)
        state["completed_models"].append(model_name)
        state["inference"][model_name] = {
            "seconds": elapsed,
            "cases_per_second": len(frame) / elapsed,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        }
        atomic_json(state, state_path)
        api.create_commit(
            repo_id=args.private_hf_repo,
            repo_type="model",
            token=token,
            operations=[
                CommitOperationAdd(
                    path_in_repo=remote_prediction,
                    path_or_fileobj=str(prediction_path),
                ),
                CommitOperationAdd(
                    path_in_repo=remote_checksum,
                    path_or_fileobj=str(checksum_path),
                ),
                CommitOperationAdd(
                    path_in_repo=f"{remote_root}/{state_path.name}",
                    path_or_fileobj=str(state_path),
                ),
            ],
            commit_message=f"recovery: save confirmation {model_name} predictions",
        )
        print(
            json.dumps(
                {
                    "event": "confirmation_model_complete",
                    "model": model_name,
                    "seconds": elapsed,
                    "confirmation_threshold_tuning": False,
                }
            )
        )
        del model, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results: dict[str, object] = {}
    for model_name in MODEL_ORDER:
        metrics = multilabel_metrics(
            probabilities[model_name], targets, thresholds=thresholds[model_name]
        )
        for label, values in zip(PRIMARY_LABELS, metrics["per_label"]):
            values["label"] = label
        results[model_name] = {
            "checkpoint_sha256": checkpoint_hashes[model_name],
            "best_validation_epoch": int(checkpoints[model_name]["epoch"]),
            "validation_macro_auroc": float(
                checkpoints[model_name]["validation_metrics"]["macro"]["auroc"]
            ),
            "validation_thresholds": thresholds[model_name].tolist(),
            "confirmation_metrics": metrics,
            "inference": state["inference"].get(model_name, {}),
        }
    bootstrap = paired_bootstrap_comparison(
        probabilities,
        targets,
        thresholds,
        reference_model="cnn",
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    summary = {
        "artifact": "Objective 2 independent confirmation comparison",
        "models": results,
        "model_order_frozen_before_confirmation": list(MODEL_ORDER),
        "reference_model": "cnn",
        "confirmation_manifest_sha256": confirmation_hash,
        "protocol_sha256": protocol_hash,
        "confirmation_cases": len(frame),
        "confirmation_patients": int(frame["patient_id"].nunique()),
        "labels": PRIMARY_LABELS,
        "bootstrap": bootstrap,
        "enhancement_developed_after_original_locked_test": True,
        "independent_confirmation_cohort": True,
        "protocol_published_before_confirmation_evaluation": True,
        "validation_thresholds_reused_without_change": True,
        "confirmation_threshold_tuning": False,
        "confirmation_used_for_model_selection": False,
        "confirmation_evaluated": True,
        "confirmation_evaluation_count": 1,
        "case_level_predictions_publication_allowed": False,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "medical_images_included": False,
        "private_manifest_included": False,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    summary_path = public_root / "independent_confirmation_summary_public.json"
    atomic_json(summary, summary_path)
    figure_path = public_root / "independent_confirmation_comparison.png"
    make_figure(summary, figure_path)
    lock = {
        "artifact": "Objective 2 immutable independent confirmation evaluation lock",
        "confirmation_manifest_sha256": confirmation_hash,
        "protocol_sha256": protocol_hash,
        "checkpoint_sha256": checkpoint_hashes,
        "summary_sha256": sha256_file(summary_path),
        "figure_sha256": sha256_file(figure_path),
        "completed_models": list(MODEL_ORDER),
        "validation_thresholds_reused_without_change": True,
        "confirmation_used_for_model_selection": False,
        "confirmation_evaluated": True,
        "confirmation_evaluation_count": 1,
    }
    atomic_json(lock, final_lock)
    api.create_commit(
        repo_id=args.private_hf_repo,
        repo_type="model",
        token=token,
        operations=[
            CommitOperationAdd(
                path_in_repo=remote_lock,
                path_or_fileobj=str(final_lock),
            ),
            CommitOperationAdd(
                path_in_repo=f"{remote_root}/{summary_path.name}",
                path_or_fileobj=str(summary_path),
            ),
            CommitOperationAdd(
                path_in_repo=f"{remote_root}/{figure_path.name}",
                path_or_fileobj=str(figure_path),
            ),
        ],
        commit_message="results: finalize independent Objective 2 confirmation",
    )

    print("--- INDEPENDENT CONFIRMATION RESULTS ---")
    for model_name in MODEL_ORDER:
        macro = results[model_name]["confirmation_metrics"]["macro"]
        ci = bootstrap["model_metric_95_ci"][model_name]
        print(
            model_name,
            "AUROC=",
            macro["auroc"],
            "95% CI=",
            ci["auroc"],
            "AUPRC=",
            macro["auprc"],
            "95% CI=",
            ci["auprc"],
            "F1=",
            macro["f1"],
            "95% CI=",
            ci["f1"],
        )
    print(
        "DenseNet minus CNN:",
        json.dumps(bootstrap["paired_model_minus_reference"]["densenet121"], indent=2),
    )
    print("Summary SHA-256:", lock["summary_sha256"])
    print("Final-lock SHA-256:", sha256_file(final_lock))
    print("Confirmation threshold tuning:", False)
    print("Confirmation used for model selection:", False)
    print("Private recovery verified:", True)
    print("OBJECTIVE 2 INDEPENDENT CONFIRMATION EVALUATION SUCCESSFUL")


if __name__ == "__main__":
    main()
