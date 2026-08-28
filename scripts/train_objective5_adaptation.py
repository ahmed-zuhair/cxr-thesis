#!/usr/bin/env python3
"""Train one locked, test-blind Objective 5 adaptation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_thesis.objective2.data import ImageClassificationDataset
from cxr_thesis.objective2.metrics import multilabel_metrics, select_f1_thresholds
from cxr_thesis.objective2.training import (
    capture_rng_state,
    optimizer_state_to_device,
    restore_rng_state,
    seed_everything,
)
from cxr_thesis.objective5.adaptation import (
    final_block_parameters,
    head_parameters,
    initialise_shared_label_densenet,
    set_adaptation_phase,
)

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Pneumothorax",
]
SOURCE_CHECKPOINT_SHA256 = (
    "2b7fa0d2f3dee3c59c538be15dd0435c71ad26b411fc1312bd7e5fe99fbac55f"
)
PROTOCOL_SHA256 = "f36064954f16f0831739cf048d223bd39aacf833cc86c3dbbde92ff3c7085dfb"
ZERO_SHOT_AUROC = {
    "chexpert": 0.7438386410545034,
    "padchest": 0.8737348139716848,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(ZERO_SHOT_AUROC), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/"))
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def manifest_labels(frame: pd.DataFrame) -> list[str]:
    casefolded = {str(column).casefold(): str(column) for column in frame.columns}
    columns = []
    for label in LABELS:
        candidates = (f"label_{label}".casefold(), label.casefold())
        matches = [
            casefolded[candidate] for candidate in candidates if candidate in casefolded
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one manifest column for {label}; found {matches}"
            )
        columns.append(matches[0])
    return columns


def validate_manifest(
    path: Path,
    expected_hash: str,
    expected_cases: int,
    expected_role: str,
) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Manifest hash mismatch: {path}")
    frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str})
    if len(frame) != expected_cases:
        raise RuntimeError(f"Expected {expected_cases} cases, found {len(frame)}")
    if "role" in frame.columns:
        observed = set(frame["role"].astype(str).str.lower())
        aliases = {
            "adaptation_train": {"adaptation_train", "adaptation", "train"},
            "target_validation": {"target_validation", "validation", "val"},
        }
        if not observed <= aliases[expected_role]:
            raise RuntimeError(f"Unexpected roles in {path}: {sorted(observed)}")
    elif "split" in frame.columns:
        expected_split = "train" if expected_role == "adaptation_train" else "val"
        if set(frame["split"].astype(str).str.lower()) != {expected_split}:
            raise RuntimeError(f"Unexpected split in {path}")
    else:
        raise ValueError("Manifest must contain role or split")
    columns = manifest_labels(frame)
    values = frame[columns].to_numpy(dtype=np.float32)
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("All target labels must be binary 0/1")
    return frame, columns


def macro_brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.mean((probabilities - targets) ** 2, axis=0)))


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 15
) -> float:
    per_label = []
    boundaries = np.linspace(0.0, 1.0, bins + 1)
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
        per_label.append(score)
    return float(np.mean(per_label))


def configure_phase(model: nn.Module, optimizer, epoch: int) -> str:
    phase = "head_warmup" if epoch <= 2 else "final_block"
    set_adaptation_phase(model, phase)
    optimizer.param_groups[0]["lr"] = 0.0 if phase == "head_warmup" else 1e-5
    optimizer.param_groups[1]["lr"] = 3e-4 if phase == "head_warmup" else 1e-4
    return phase


def train_epoch(model, loader, optimizer, criterion, device, phase: str) -> float:
    model.train()
    model.encoder.eval()
    if phase == "final_block":
        model.encoder.features.denseblock4.train()
        model.encoder.features.norm5.train()
    total_loss = 0.0
    total_cases = 0
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    batches = len(loader)
    for batch_index, batch in enumerate(loader):
        offset = batch_index % 2
        group_size = min(2, batches - (batch_index - offset))
        image = batch["image"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = criterion(model(image, clinical), labels)
        scaler.scale(loss / group_size).backward()
        if offset + 1 == group_size:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * int(labels.shape[0])
        total_cases += int(labels.shape[0])
    return total_loss / max(1, total_cases)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, targets = [], []
    for batch in loader:
        logits = model(
            batch["image"].to(device, non_blocking=True),
            batch["clinical"].to(device, non_blocking=True),
        )
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["labels"].numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def upload_files(api, CommitOperationAdd, args, names, message: str) -> None:
    paths = [args.output_dir / name for name in names]
    paths = [path for path in paths if path.is_file()]
    api.create_commit(
        repo_id=args.hf_repo,
        repo_type="model",
        token=os.environ["HF_TOKEN"],
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.hf_path.strip('/')}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in paths
        ],
        commit_message=message,
    )


def restore_remote(api, hf_hub_download, args) -> bool:
    prefix = args.hf_path.strip("/")
    remote = set(
        api.list_repo_files(
            args.hf_repo, repo_type="model", token=os.environ["HF_TOKEN"]
        )
    )
    names = (
        "last.pt",
        "last.sha256",
        "history_progress.csv",
        "best.pt",
        "best.sha256",
        "history.csv",
        "validation_summary.json",
    )
    found = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        remote_name = f"{prefix}/{name}"
        if remote_name not in remote:
            continue
        downloaded = Path(
            hf_hub_download(
                args.hf_repo,
                filename=remote_name,
                repo_type="model",
                token=os.environ["HF_TOKEN"],
                force_download=True,
            )
        )
        shutil.copy2(downloaded, args.output_dir / name)
        found = True
    for checkpoint_name, checksum_name in (
        ("last.pt", "last.sha256"),
        ("best.pt", "best.sha256"),
    ):
        checkpoint = args.output_dir / checkpoint_name
        checksum = args.output_dir / checksum_name
        if checkpoint.is_file() and checksum.is_file():
            expected = checksum.read_text(encoding="utf-8").split()[0]
            if sha256_file(checkpoint) != expected:
                raise RuntimeError(f"Recovered {checkpoint_name} hash mismatch")
    return found


def main() -> None:
    args = parse_args()
    if not os.environ.get("HF_TOKEN", "").strip():
        raise RuntimeError("HF_TOKEN is not loaded")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=os.environ["HF_TOKEN"])
    if not bool(api.model_info(args.hf_repo, token=os.environ["HF_TOKEN"]).private):
        raise RuntimeError("The recovery repository must remain private")

    train_frame, train_columns = validate_manifest(
        args.train_manifest, args.expected_train_sha256, 30_000, "adaptation_train"
    )
    val_frame, val_columns = validate_manifest(
        args.val_manifest, args.expected_val_sha256, 5_000, "target_validation"
    )
    if set(train_frame["patient_id"]) & set(val_frame["patient_id"]):
        raise RuntimeError("Patient leakage exists between adaptation and validation")
    if sha256_file(args.source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("Confirmed DenseNet checkpoint hash mismatch")

    restored = restore_remote(api, hf_hub_download, args)
    restored = restored or (args.output_dir / "last.pt").is_file()
    final_summary_path = args.output_dir / "validation_summary.json"
    if final_summary_path.is_file():
        summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("dataset") != args.dataset
            or summary.get("test_evaluated") is not False
        ):
            raise RuntimeError("Recovered final result does not match this run")
        print(
            json.dumps(
                {
                    "event": "final_adaptation_restored",
                    "dataset": args.dataset,
                    "training_repeated": False,
                    "test_evaluated": False,
                }
            )
        )
        print("OBJECTIVE 5 PRIVATE ADAPTATION RESULT RESTORED SUCCESSFULLY")
        return

    seed_everything(42)
    source_checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    if source_checkpoint.get("test_evaluated") is not False:
        raise RuntimeError("Source checkpoint is not test-blind")
    model, source_indices = initialise_shared_label_densenet(source_checkpoint, LABELS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = ImageClassificationDataset(
        train_frame,
        train_columns,
        data_root=args.data_root,
        image_size=320,
        augment=True,
        seed=42,
        augmentation_profile="objective5_locked",
        epoch_varying_augmentation=True,
        output_channels=3,
        normalisation="imagenet",
        horizontal_flip_probability=0.0,
    )
    val_dataset = ImageClassificationDataset(
        val_frame,
        val_columns,
        data_root=args.data_root,
        image_size=320,
        augment=False,
        seed=42,
        output_channels=3,
        normalisation="imagenet",
        horizontal_flip_probability=0.0,
    )
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    positives = train_frame[train_columns].sum(axis=0).to_numpy(dtype=np.float32)
    negatives = len(train_frame) - positives
    positive_weights = np.minimum(np.sqrt(negatives / np.maximum(positives, 1.0)), 10.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": final_block_parameters(model), "lr": 0.0},
            {"params": head_parameters(model), "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    signature = {
        "artifact": "Objective 5 test-blind target-domain adaptation",
        "dataset": args.dataset,
        "labels": LABELS,
        "source_label_indices": source_indices,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "train_manifest_sha256": args.expected_train_sha256,
        "val_manifest_sha256": args.expected_val_sha256,
        "train_cases": 30_000,
        "validation_cases": 5_000,
        "seed": 42,
        "image_size": 320,
        "batch_size": 16,
        "accumulation_steps": 2,
        "maximum_epochs": 12,
        "warmup_epochs": 2,
        "patience": 3,
        "test_cases_accessed": 0,
    }
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    history = []
    best_auroc = -np.inf
    stale = 0
    start_epoch = 1
    resume_count = 0
    if restored and last_path.is_file():
        last_checksum = args.output_dir / "last.sha256"
        if not last_checksum.is_file():
            raise FileNotFoundError("Recovery last.sha256 is missing")
        recorded_last_hash = last_checksum.read_text(encoding="utf-8").split()[0]
        if sha256_file(last_path) != recorded_last_hash:
            raise RuntimeError("Recovery last.pt hash mismatch")
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if (
            state.get("signature") != signature
            or state.get("test_evaluated") is not False
        ):
            raise RuntimeError("Recovered training state does not match the locked run")
        if sha256_file(best_path) != state.get("best_checkpoint_sha256"):
            raise RuntimeError(
                "Recovered best.pt does not match the epoch recovery state"
            )
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        optimizer_state_to_device(optimizer, device)
        generator.set_state(state["generator_state"].cpu())
        restore_rng_state(state["rng_state"])
        history = list(state["history"])
        best_auroc = float(state["best_auroc"])
        stale = int(state["stale"])
        start_epoch = int(state["epoch_completed"]) + 1
        resume_count = int(state.get("resume_count", 0)) + 1
        print(
            json.dumps(
                {
                    "resume": True,
                    "completed_epochs": start_epoch - 1,
                    "next_epoch": start_epoch,
                    "resume_count": resume_count,
                }
            )
        )

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "train_cases": len(train_dataset),
                "validation_cases": len(val_dataset),
                "labels": LABELS,
                "device": str(device),
                "test_cases_accessed": 0,
            },
            indent=2,
        )
    )
    if start_epoch > 2 and stale >= 3:
        start_epoch = 13
    for epoch in range(start_epoch, 13):
        train_dataset.set_epoch(epoch)
        phase = configure_phase(model, optimizer, epoch)
        loss = train_epoch(model, train_loader, optimizer, criterion, device, phase)
        probabilities, targets = predict(model, val_loader, device)
        metrics = multilabel_metrics(probabilities, targets, thresholds=0.5)
        macro_auroc = float(metrics["macro"]["auroc"])
        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": loss,
            "validation_macro_auroc": macro_auroc,
            "validation_macro_auprc": float(metrics["macro"]["auprc"]),
            "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if np.isfinite(macro_auroc) and macro_auroc > best_auroc:
            best_auroc = macro_auroc
            stale = 0
            atomic_torch_save(
                {
                    "model_name": "densenet121",
                    "model_state": model.state_dict(),
                    "label_names": LABELS,
                    "epoch": epoch,
                    "validation_macro_auroc": macro_auroc,
                    "seed": 42,
                    "model_config": {
                        "image_size": 320,
                        "clinical_dim": 9,
                        "input_channels": 3,
                        "normalisation": "imagenet",
                        "dropout": 0.2,
                    },
                    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                    "protocol_sha256": PROTOCOL_SHA256,
                    "dataset": args.dataset,
                    "test_evaluated": False,
                },
                best_path,
            )
        elif epoch > 2:
            stale += 1
        pd.DataFrame(history).to_csv(
            args.output_dir / "history_progress.csv", index=False
        )
        best_hash = sha256_file(best_path)
        state = {
            "format_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "rng_state": capture_rng_state(),
            "epoch_completed": epoch,
            "best_auroc": best_auroc,
            "stale": stale,
            "history": history,
            "signature": signature,
            "best_checkpoint_sha256": best_hash,
            "resume_count": resume_count,
            "test_evaluated": False,
        }
        atomic_torch_save(state, last_path)
        last_hash = sha256_file(last_path)
        (args.output_dir / "last.sha256").write_text(
            f"{last_hash}  last.pt\n", encoding="utf-8"
        )
        upload_files(
            api,
            CommitOperationAdd,
            args,
            ["last.pt", "last.sha256", "history_progress.csv", "best.pt"],
            f"recovery: Objective 5 {args.dataset} epoch {epoch}",
        )
        print(
            json.dumps(
                {
                    "private_recovery_uploaded_epoch": epoch,
                    "dataset": args.dataset,
                    "test_evaluated": False,
                }
            ),
            flush=True,
        )
        if epoch > 2 and stale >= 3:
            break

    if not best_path.is_file():
        raise RuntimeError("Training did not produce a validation checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    probabilities, targets = predict(model, val_loader, device)
    thresholds = select_f1_thresholds(probabilities, targets)
    final_metrics = multilabel_metrics(probabilities, targets, thresholds=thresholds)
    for label, entry in zip(LABELS, final_metrics["per_label"]):
        entry["label"] = label
    adapted_auroc = float(final_metrics["macro"]["auroc"])
    reference_auroc = ZERO_SHOT_AUROC[args.dataset]
    advances = adapted_auroc >= reference_auroc + 0.005
    best.update(
        {
            "validation_thresholds": thresholds.tolist(),
            "validation_metrics": json_safe(final_metrics),
            "validation_macro_brier": macro_brier(probabilities, targets),
            "validation_macro_ece": expected_calibration_error(probabilities, targets),
            "positive_weights": positive_weights.tolist(),
            "training_signature": signature,
            "candidate_advances": bool(advances),
            "zero_shot_reference_auroc": reference_auroc,
            "minimum_advancement_delta": 0.005,
            "test_evaluated": False,
        }
    )
    atomic_torch_save(best, best_path)
    best_hash = sha256_file(best_path)
    (args.output_dir / "best.sha256").write_text(
        f"{best_hash}  best.pt\n", encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    summary = {
        "artifact": "Objective 5 target-domain adaptation validation candidate",
        "dataset": args.dataset,
        "model": "densenet121",
        "labels": LABELS,
        "training_cases": 30_000,
        "validation_cases": 5_000,
        "best_epoch": int(best["epoch"]),
        "validation_metrics": json_safe(final_metrics),
        "validation_macro_brier": best["validation_macro_brier"],
        "validation_macro_ece": best["validation_macro_ece"],
        "zero_shot_reference_auroc": reference_auroc,
        "adapted_minus_zero_shot_auroc": adapted_auroc - reference_auroc,
        "minimum_advancement_delta": 0.005,
        "candidate_advances": bool(advances),
        "selected_candidate": "adapted" if advances else "zero_shot",
        "checkpoint_sha256": best_hash,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "resume_count": resume_count,
        "test_cases_accessed": 0,
        "test_evaluated": False,
    }
    final_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    upload_files(
        api,
        CommitOperationAdd,
        args,
        [
            "best.pt",
            "best.sha256",
            "history.csv",
            "validation_summary.json",
            "last.pt",
            "last.sha256",
            "history_progress.csv",
        ],
        f"recovery: finalize Objective 5 {args.dataset} adaptation",
    )
    print(
        json.dumps(
            {
                "event": "private_adaptation_finalized",
                "dataset": args.dataset,
                "best_epoch": summary["best_epoch"],
                "validation_macro_auroc": adapted_auroc,
                "zero_shot_macro_auroc": reference_auroc,
                "adapted_minus_zero_shot": summary["adapted_minus_zero_shot_auroc"],
                "candidate_advances": advances,
                "checkpoint_sha256": best_hash,
                "test_evaluated": False,
                "private_recovery_verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 5 TEST-BLIND ADAPTATION WITH PRIVATE RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()
