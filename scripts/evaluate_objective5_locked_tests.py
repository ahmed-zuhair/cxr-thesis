#!/usr/bin/env python3
"""Evaluate the two frozen Objective 5 candidates once on their locked tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_thesis.objective2.data import ImageClassificationDataset
from cxr_thesis.objective2.evaluation import percentile_interval
from cxr_thesis.objective2.metrics import multilabel_metrics
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective2.training import seed_everything


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Pneumothorax",
]
DATASETS = ("chexpert", "padchest")
EXPECTED = {
    "chexpert": {
        "cases": 200,
        "patients": 200,
        "manifest_sha256": "795d538a5bfc501049ffba7611dfb2757741866c41ccbb1586f5d035a77da665",
        "checkpoint_sha256": "edcd5792c57f04bdbef88043a2a11e422b506bdc2f26cd96f13121f6a8029c12",
    },
    "padchest": {
        "cases": 5_000,
        "patients": 5_000,
        "manifest_sha256": "8b07c7cae5671af072ae3e39fdd5380e6e9741521e214a7ad93d9bd22a2151c6",
        "checkpoint_sha256": "109db89a723c6e2f24442cb5866bfcf4084e85083936cda91bce3b8ae4365d9d",
    },
}
PROTOCOL_SHA256 = "f36064954f16f0831739cf048d223bd39aacf833cc86c3dbbde92ff3c7085dfb"
SELECTION_SUMMARY_SHA256 = "aaedfd9f894c0842b87ada18653968f59ab7e1d6e932c0bd8e7a7826ef55196b"
SELECTION_LOCK_SHA256 = "1981344a8d1a0c40bd8b5245f3750738e88e6b6933292f4525937335511c470f"
FINAL_LOCK_NAME = "FINAL_OBJECTIVE5_LOCKED_TEST_EVALUATION.json"
SUMMARY_NAME = "objective5_external_locked_test_summary_public.json"
FIGURE_NAME = "objective5_external_locked_test_metrics.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in DATASETS:
        parser.add_argument(f"--{dataset}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{dataset}-test", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--protocol-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--private-hf-repo", required=True)
    parser.add_argument("--private-hf-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
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
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def label_columns(frame: pd.DataFrame) -> list[str]:
    available = {str(column).casefold(): str(column) for column in frame.columns}
    columns = []
    for label in LABELS:
        candidates = (f"label_{label}".casefold(), label.casefold())
        matches = [available[value] for value in candidates if value in available]
        if len(matches) != 1:
            raise ValueError(f"Expected one locked-test column for {label}: {matches}")
        columns.append(matches[0])
    return columns


def macro_brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.mean((probabilities - targets) ** 2, axis=0)))


def macro_ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    scores = []
    for label in range(probabilities.shape[1]):
        score = 0.0
        for index in range(bins):
            lower, upper = boundaries[index : index + 2]
            selected = (probabilities[:, label] >= lower) & (
                probabilities[:, label] < upper
                if index < bins - 1
                else probabilities[:, label] <= upper
            )
            if selected.any():
                score += float(selected.mean()) * abs(
                    float(probabilities[selected, label].mean())
                    - float(targets[selected, label].mean())
                )
        scores.append(score)
    return float(np.mean(scores))


def metric_bundle(
    probabilities: np.ndarray, targets: np.ndarray, thresholds: np.ndarray
) -> dict[str, object]:
    result = multilabel_metrics(probabilities, targets, thresholds=thresholds)
    result["macro"]["brier"] = macro_brier(probabilities, targets)
    result["macro"]["ece"] = macro_ece(probabilities, targets)
    for label, entry in zip(LABELS, result["per_label"]):
        entry["label"] = label
        entry["positive_count"] = int(targets[:, LABELS.index(label)].sum())
    return result


def patient_bootstrap(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
    patient_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Patient-cluster bootstrap; locked cohorts contain one image per patient."""

    unique_patients = pd.unique(patient_ids)
    patient_rows = {
        patient: np.flatnonzero(patient_ids == patient) for patient in unique_patients
    }
    macro_names = ("auroc", "auprc", "f1", "brier", "ece")
    per_label_names = ("auroc", "auprc", "f1")
    macro = {name: np.full(replicates, np.nan) for name in macro_names}
    per_label = {
        label: {name: np.full(replicates, np.nan) for name in per_label_names}
        for label in LABELS
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled_patients = rng.choice(
            unique_patients, size=len(unique_patients), replace=True
        )
        indices = np.concatenate([patient_rows[value] for value in sampled_patients])
        sampled_probabilities = probabilities[indices]
        sampled_targets = targets[indices]
        metrics = metric_bundle(sampled_probabilities, sampled_targets, thresholds)
        for name in macro_names:
            macro[name][replicate] = float(metrics["macro"][name])
        for label_index, label in enumerate(LABELS):
            for name in per_label_names:
                per_label[label][name][replicate] = float(
                    metrics["per_label"][label_index][name]
                )
    return {
        "method": "patient-cluster percentile bootstrap",
        "replicates": replicates,
        "seed": seed,
        "macro_95_ci": {
            name: percentile_interval(values) for name, values in macro.items()
        },
        "macro_valid_replicates": {
            name: int(np.isfinite(values).sum()) for name, values in macro.items()
        },
        "per_label_95_ci": {
            label: {
                name: percentile_interval(values)
                for name, values in distributions.items()
            }
            for label, distributions in per_label.items()
        },
        "per_label_valid_replicates": {
            label: {
                name: int(np.isfinite(values).sum())
                for name, values in distributions.items()
            }
            for label, distributions in per_label.items()
        },
    }


@torch.no_grad()
def infer_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for batch in loader:
        output = model(
            batch["image"].to(device, non_blocking=True),
            batch["clinical"].to(device, non_blocking=True),
        )
        logits.append(output.float().cpu().numpy())
        targets.append(batch["labels"].numpy())
    return np.concatenate(logits), np.concatenate(targets)


def validate_checkpoint(path: Path, dataset: str) -> dict[str, object]:
    expected = EXPECTED[dataset]["checkpoint_sha256"]
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"{dataset} checkpoint SHA-256 does not match")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != "densenet121":
        raise RuntimeError(f"{dataset} candidate is not DenseNet-121")
    if checkpoint.get("label_names") != LABELS:
        raise RuntimeError(f"{dataset} checkpoint label order changed")
    if checkpoint.get("test_evaluated") is not False:
        raise RuntimeError(f"{dataset} checkpoint is not test-blind")
    if "model_state" not in checkpoint:
        raise RuntimeError(f"{dataset} checkpoint has no model state")
    return checkpoint


def validate_locks(args: argparse.Namespace) -> dict[str, object]:
    checks = {
        args.selection_summary: SELECTION_SUMMARY_SHA256,
        args.selection_lock: SELECTION_LOCK_SHA256,
        args.protocol_record: PROTOCOL_SHA256,
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Protected lock hash does not match: {path}")
    selection = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if lock.get("summary_sha256") != SELECTION_SUMMARY_SHA256:
        raise RuntimeError("Selection lock does not identify the frozen summary")
    if selection.get("status") != "locked before either external-domain test evaluation":
        raise RuntimeError("Selection summary is not a pre-test lock")
    if selection.get("test_evaluated") is not False:
        raise RuntimeError("Selection summary is not test-blind")
    for dataset in DATASETS:
        record = selection.get("datasets", {}).get(dataset, {})
        thresholds = np.asarray(record.get("frozen_thresholds"), dtype=float)
        temperature = float(record.get("temperature", np.nan))
        if record.get("selected_candidate") != "adapted":
            raise RuntimeError(f"{dataset} selected candidate is not adapted")
        if record.get("candidate_checkpoint_sha256") != EXPECTED[dataset]["checkpoint_sha256"]:
            raise RuntimeError(f"{dataset} selected checkpoint changed")
        if thresholds.shape != (len(LABELS),) or not np.isfinite(thresholds).all():
            raise RuntimeError(f"{dataset} frozen thresholds are invalid")
        if not np.isfinite(temperature) or temperature <= 0:
            raise RuntimeError(f"{dataset} frozen temperature is invalid")
    return selection


def make_figure(summary: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    names = ["CheXpert", "PadChest"]
    metrics = ("auroc", "auprc", "f1", "brier", "ece")
    titles = ("Macro AUROC", "Macro AUPRC", "Macro F1", "Macro Brier", "Macro ECE")
    figure, axes = plt.subplots(1, 5, figsize=(18, 4.3))
    for axis, metric, title in zip(axes, metrics, titles):
        values = [summary["datasets"][name]["test_metrics"]["macro"][metric] for name in DATASETS]
        intervals = [summary["datasets"][name]["bootstrap"]["macro_95_ci"][metric] for name in DATASETS]
        errors = np.asarray(
            [
                [value - interval[0] for value, interval in zip(values, intervals)],
                [interval[1] - value for value, interval in zip(values, intervals)],
            ]
        )
        axis.bar(names, values, color=["#3569a8", "#d67835"])
        axis.errorbar(range(2), values, yerr=errors, fmt="none", ecolor="black", capsize=4)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Objective 5 External-Domain Locked-Test Evaluation")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "cxr-thesis"})
    plt.close(figure)


def upload_private(api, repo: str, remote_root: str, paths: list[Path], token: str) -> None:
    from huggingface_hub import CommitOperationAdd, hf_hub_download

    remote_files = set(api.list_repo_files(repo, repo_type="model", token=token))
    operations = []
    for path in paths:
        remote = f"{remote_root}/{path.name}"
        if remote in remote_files:
            downloaded = Path(
                hf_hub_download(
                    repo,
                    filename=remote,
                    repo_type="model",
                    token=token,
                    force_download=True,
                )
            )
            if sha256_file(downloaded) != sha256_file(path):
                raise RuntimeError(f"Existing private recovery artifact differs: {remote}")
            continue
        operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)))
    if operations:
        api.create_commit(
            repo_id=repo,
            repo_type="model",
            token=token,
            operations=operations,
            commit_message="private recovery: Objective 5 locked-test evaluation",
        )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.bootstrap_replicates <= 0:
        raise ValueError("Batch size/replicates must be positive and workers nonnegative")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for private recovery")
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    if not bool(api.model_info(args.private_hf_repo, token=token).private):
        raise RuntimeError("Objective 5 recovery repository must remain private")
    remote_root = args.private_hf_path.strip("/")
    remote_files = set(api.list_repo_files(args.private_hf_repo, repo_type="model", token=token))

    # Every frozen decision is authenticated before any locked-test manifest is opened.
    selection = validate_locks(args)
    checkpoints = {
        dataset: validate_checkpoint(getattr(args, f"{dataset}_checkpoint"), dataset)
        for dataset in DATASETS
    }
    final_remote = f"{remote_root}/{FINAL_LOCK_NAME}"
    if final_remote in remote_files:
        restore_names = [FINAL_LOCK_NAME, SUMMARY_NAME, f"{SUMMARY_NAME}.sha256", FIGURE_NAME, f"{FIGURE_NAME}.sha256"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in restore_names:
            remote = f"{remote_root}/{name}"
            if remote not in remote_files:
                raise RuntimeError("Private finalized recovery is incomplete")
            downloaded = Path(hf_hub_download(args.private_hf_repo, filename=remote, repo_type="model", token=token, force_download=True))
            target = args.output_dir / name if name == FINAL_LOCK_NAME else args.output_dir / "public" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(downloaded.read_bytes())
        final_lock = json.loads((args.output_dir / FINAL_LOCK_NAME).read_text(encoding="utf-8"))
        if final_lock.get("test_evaluation_count_per_dataset") != 1 or final_lock.get("test_evaluated") is not True:
            raise RuntimeError("Recovered Objective 5 final lock is invalid")
        if final_lock.get("summary_sha256") != sha256_file(args.output_dir / "public" / SUMMARY_NAME):
            raise RuntimeError("Recovered Objective 5 summary hash does not match")
        print("OBJECTIVE 5 FINAL LOCKED-TEST EVALUATION RESTORED WITHOUT RE-EVALUATION")
        return

    final_lock_path = args.output_dir / FINAL_LOCK_NAME
    if final_lock_path.is_file():
        public_root = args.output_dir / "public"
        summary_path = public_root / SUMMARY_NAME
        summary_checksum = public_root / f"{SUMMARY_NAME}.sha256"
        figure_path = public_root / FIGURE_NAME
        figure_checksum = public_root / f"{FIGURE_NAME}.sha256"
        for path in (
            summary_path,
            summary_checksum,
            figure_path,
            figure_checksum,
        ):
            if not path.is_file():
                raise RuntimeError("Local finalized Objective 5 output is incomplete")
        local_lock = json.loads(final_lock_path.read_text(encoding="utf-8"))
        local_checks = {
            "evaluated": local_lock.get("test_evaluated") is True,
            "count": local_lock.get("test_evaluation_count_per_dataset") == 1,
            "datasets": local_lock.get("completed_datasets") == list(DATASETS),
            "summary": local_lock.get("summary_sha256") == sha256_file(summary_path),
            "figure": local_lock.get("figure_sha256") == sha256_file(figure_path),
        }
        if not all(local_checks.values()):
            raise RuntimeError(f"Local finalized Objective 5 lock is invalid: {local_checks}")
        upload_private(
            api,
            args.private_hf_repo,
            remote_root,
            [
                summary_path,
                summary_checksum,
                figure_path,
                figure_checksum,
                final_lock_path,
            ],
            token,
        )
        print("OBJECTIVE 5 LOCAL FINAL EVALUATION BACKED UP WITHOUT RE-EVALUATION")
        return
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume only after interruption")
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError("Resume output directory does not exist")

    # Locked-test data access begins only here.
    frames: dict[str, pd.DataFrame] = {}
    targets: dict[str, np.ndarray] = {}
    order_hashes: dict[str, str] = {}
    for dataset in DATASETS:
        path = getattr(args, f"{dataset}_test")
        expected = EXPECTED[dataset]
        if not path.is_file() or sha256_file(path) != expected["manifest_sha256"]:
            raise RuntimeError(f"{dataset} locked-test manifest SHA-256 does not match")
        frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str})
        if len(frame) != expected["cases"] or frame["patient_id"].nunique() != expected["patients"]:
            raise RuntimeError(f"{dataset} locked-test size does not match")
        if frame["patient_id"].duplicated().any() or frame["image_id"].duplicated().any():
            raise RuntimeError(f"{dataset} locked test is not one image per patient")
        if "role" in frame.columns:
            roles = set(frame["role"].astype(str).str.lower())
            if not roles <= {"locked_test", "test"}:
                raise RuntimeError(f"{dataset} manifest contains a non-test role")
        elif "split" in frame.columns and set(frame["split"].astype(str).str.lower()) != {"test"}:
            raise RuntimeError(f"{dataset} manifest contains a non-test split")
        columns = label_columns(frame)
        values = frame[columns].to_numpy(dtype=np.int8)
        if not np.isin(values, [0, 1]).all():
            raise RuntimeError(f"{dataset} locked-test labels are not binary")
        frames[dataset] = frame
        targets[dataset] = values
        order_hashes[dataset] = sha256_text("\n".join(frame["image_id"].astype(str)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_root = args.output_dir / "private"
    public_root = args.output_dir / "public"
    private_root.mkdir(exist_ok=True)
    public_root.mkdir(exist_ok=True)
    state_path = private_root / "objective5_locked_test_state_private.json"
    signature = {
        "format_version": 1,
        "manifest_sha256": {name: EXPECTED[name]["manifest_sha256"] for name in DATASETS},
        "checkpoint_sha256": {name: EXPECTED[name]["checkpoint_sha256"] for name in DATASETS},
        "selection_summary_sha256": SELECTION_SUMMARY_SHA256,
        "selection_lock_sha256": SELECTION_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "case_order_sha256": order_hashes,
        "seed": args.seed,
    }
    state = {"signature": signature, "completed_datasets": []}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise RuntimeError("Interrupted Objective 5 state signature does not match")
    else:
        atomic_json(state, state_path)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        prediction_path = private_root / f"{dataset}_locked_test_predictions_private.npz"
        checksum_path = prediction_path.with_suffix(".npz.sha256")
        remote_prediction = f"{remote_root}/{prediction_path.name}"
        remote_checksum = f"{remote_root}/{checksum_path.name}"
        if not prediction_path.is_file() and remote_prediction in remote_files:
            if remote_checksum not in remote_files:
                raise RuntimeError(f"Incomplete private recovery for {dataset}")
            downloaded_checksum = Path(hf_hub_download(args.private_hf_repo, filename=remote_checksum, repo_type="model", token=token, force_download=True))
            expected_hash = downloaded_checksum.read_text(encoding="utf-8").split()[0]
            downloaded = Path(hf_hub_download(args.private_hf_repo, filename=remote_prediction, repo_type="model", token=token, force_download=True))
            if sha256_file(downloaded) != expected_hash:
                raise RuntimeError(f"Private {dataset} prediction hash does not match")
            prediction_path.write_bytes(downloaded.read_bytes())
            checksum_path.write_text(f"{expected_hash}  {prediction_path.name}\n", encoding="utf-8")
        if prediction_path.is_file():
            saved = np.load(prediction_path)
            restored = np.asarray(saved["probabilities"], dtype=np.float32)
            if restored.shape != targets[dataset].shape or not np.array_equal(saved["targets"], targets[dataset]):
                raise RuntimeError(f"Restored {dataset} predictions are incompatible")
            if str(saved["case_order_sha256"].item()) != order_hashes[dataset]:
                raise RuntimeError(f"Restored {dataset} case order changed")
            probabilities[dataset] = restored
            print(json.dumps({"event": "private_predictions_restored", "dataset": dataset}))
            continue

        record = selection["datasets"][dataset]
        frame = frames[dataset]
        columns = label_columns(frame)
        dataset_object = ImageClassificationDataset(
            frame,
            columns,
            data_root=args.data_root,
            image_size=320,
            augment=False,
            output_channels=3,
            normalisation="imagenet",
            horizontal_flip_probability=0.0,
        )
        loader = DataLoader(
            dataset_object,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
        )
        model = build_classifier("densenet121", len(LABELS), image_size=320, pretrained=False, dropout=0.2)
        model.load_state_dict(checkpoints[dataset]["model_state"], strict=True)
        model.to(device)
        logits, observed_targets = infer_logits(model, loader, device)
        if not np.array_equal(observed_targets.astype(np.int8), targets[dataset]):
            raise RuntimeError(f"{dataset} inference label order changed")
        temperature = float(record["temperature"])
        result = 1.0 / (1.0 + np.exp(-(logits / temperature)))
        probabilities[dataset] = result.astype(np.float32)
        atomic_npz(
            prediction_path,
            probabilities=probabilities[dataset],
            targets=targets[dataset],
            case_order_sha256=np.asarray(order_hashes[dataset]),
        )
        prediction_hash = sha256_file(prediction_path)
        checksum_path.write_text(f"{prediction_hash}  {prediction_path.name}\n", encoding="utf-8")
        if dataset not in state["completed_datasets"]:
            state["completed_datasets"].append(dataset)
        atomic_json(state, state_path)
        upload_private(api, args.private_hf_repo, remote_root, [prediction_path, checksum_path, state_path], token)
        print(json.dumps({"event": "locked_test_inference_saved", "dataset": dataset, "cases": len(frame)}))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dataset_results = {}
    for index, dataset in enumerate(DATASETS):
        record = selection["datasets"][dataset]
        thresholds = np.asarray(record["frozen_thresholds"], dtype=float)
        metrics = metric_bundle(probabilities[dataset], targets[dataset], thresholds)
        bootstrap = patient_bootstrap(
            probabilities[dataset],
            targets[dataset],
            thresholds,
            frames[dataset]["patient_id"].astype(str).to_numpy(),
            replicates=args.bootstrap_replicates,
            seed=args.seed + index * 1_000,
        )
        dataset_results[dataset] = {
            "selected_candidate": "adapted DenseNet-121",
            "test_cases": int(len(frames[dataset])),
            "test_patients": int(frames[dataset]["patient_id"].nunique()),
            "positive_counts": {label: int(targets[dataset][:, label_index].sum()) for label_index, label in enumerate(LABELS)},
            "checkpoint_sha256": EXPECTED[dataset]["checkpoint_sha256"],
            "locked_test_manifest_sha256": EXPECTED[dataset]["manifest_sha256"],
            "frozen_temperature": float(record["temperature"]),
            "frozen_thresholds": thresholds.tolist(),
            "test_metrics": metrics,
            "bootstrap": bootstrap,
            "low_prevalence_warning": "Pneumothorax has six positives; label-level estimates and intervals are unstable.",
        }

    summary = {
        "artifact": "Objective 5 final external-domain locked-test evaluation",
        "version": "1.0.0",
        "datasets": dataset_results,
        "labels": LABELS,
        "primary_metric": "macro AUROC",
        "secondary_metrics": ["macro AUPRC", "macro F1", "macro Brier score", "macro ECE"],
        "selection_summary_sha256": SELECTION_SUMMARY_SHA256,
        "selection_lock_sha256": SELECTION_LOCK_SHA256,
        "adaptation_protocol_sha256": PROTOCOL_SHA256,
        "temperatures_reused_without_change": True,
        "thresholds_reused_without_change": True,
        "test_threshold_tuning": False,
        "test_temperature_tuning": False,
        "test_used_for_model_selection": False,
        "test_evaluation_count_per_dataset": 1,
        "test_evaluated": True,
        "patient_identifiers_published": False,
        "image_identifiers_published": False,
        "medical_images_published": False,
        "case_level_predictions_published": False,
        "private_manifests_published": False,
    }
    summary_path = public_root / SUMMARY_NAME
    atomic_json(summary, summary_path)
    summary_hash = sha256_file(summary_path)
    summary_checksum = public_root / f"{SUMMARY_NAME}.sha256"
    summary_checksum.write_text(f"{summary_hash}  {SUMMARY_NAME}\n", encoding="utf-8")
    figure_path = public_root / FIGURE_NAME
    make_figure(summary, figure_path)
    figure_hash = sha256_file(figure_path)
    figure_checksum = public_root / f"{FIGURE_NAME}.sha256"
    figure_checksum.write_text(f"{figure_hash}  {FIGURE_NAME}\n", encoding="utf-8")
    final_lock = {
        "artifact": "Final immutable Objective 5 locked-test evaluation lock",
        "version": "1.0.0",
        "manifest_sha256": {name: EXPECTED[name]["manifest_sha256"] for name in DATASETS},
        "checkpoint_sha256": {name: EXPECTED[name]["checkpoint_sha256"] for name in DATASETS},
        "selection_summary_sha256": SELECTION_SUMMARY_SHA256,
        "selection_lock_sha256": SELECTION_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "summary_sha256": summary_hash,
        "figure_sha256": figure_hash,
        "completed_datasets": list(DATASETS),
        "temperatures_reused_without_change": True,
        "thresholds_reused_without_change": True,
        "test_used_for_model_selection": False,
        "test_evaluation_count_per_dataset": 1,
        "test_evaluated": True,
        "immutable": True,
    }
    atomic_json(final_lock, final_lock_path)
    upload_private(
        api,
        args.private_hf_repo,
        remote_root,
        [state_path, summary_path, summary_checksum, figure_path, figure_checksum, final_lock_path],
        token,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n--- FINAL OBJECTIVE 5 LOCKED-TEST STATUS ---")
    print("Summary SHA-256:", summary_hash)
    print("Final-lock SHA-256:", sha256_file(final_lock_path))
    print("Test evaluation count per dataset: 1")
    print("Threshold tuning on test:", False)
    print("Temperature tuning on test:", False)
    print("Private recovery verified:", True)
    print("OBJECTIVE 5 SINGLE LOCKED-TEST EVALUATION SUCCESSFUL")


if __name__ == "__main__":
    main()
