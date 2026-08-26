#!/usr/bin/env python3
"""Train one frozen Objective 2 model using training and validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_thesis.objective2.data import (
    GraphClassificationDataset,
    ImageClassificationDataset,
    collate_graph_samples,
)
from cxr_thesis.objective2.metrics import multilabel_metrics, select_f1_thresholds
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective2.training import (
    optimizer_state_to_device,
    predict,
    restore_rng_state,
    save_checkpoint,
    save_training_state,
    seed_everything,
    train_epoch,
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Objective 2 classifier without accessing the test cohort."
    )
    parser.add_argument("--model", required=True, choices=["cnn", "attention_cnn", "vit", "gcn", "gat"])
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/last.pt after a completed epoch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(f"Resume output directory does not exist: {args.output_dir}")
    if args.resume and (args.output_dir / "validation_summary.json").is_file():
        raise RuntimeError("Training is already complete; refusing to resume it")
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch-size must be positive")
    graph_model = args.model in {"gcn", "gat"}
    if graph_model and args.graph_root is None:
        raise ValueError("--graph-root is required for GCN and GAT")
    train_frame = pd.read_csv(args.train_manifest)
    validation_frame = pd.read_csv(args.val_manifest)
    for name, frame, expected_split in (
        ("training", train_frame, "train"),
        ("validation", validation_frame, "val"),
    ):
        observed = set(frame["split"].astype(str).str.lower())
        if observed != {expected_split}:
            raise ValueError(f"{name} manifest has unexpected splits: {sorted(observed)}")
        if "test" in observed:
            raise ValueError("Test rows are forbidden during model training")
    if args.limit_train is not None:
        train_frame = train_frame.iloc[: args.limit_train].copy()
    if args.limit_val is not None:
        validation_frame = validation_frame.iloc[: args.limit_val].copy()
    label_columns = [f"label_{label}" for label in PRIMARY_LABELS]
    missing = sorted(set(label_columns) - set(train_frame.columns))
    if missing:
        raise ValueError(f"Training labels are missing: {missing}")
    seed_everything(args.seed)
    if graph_model:
        train_dataset = GraphClassificationDataset(train_frame, label_columns, args.graph_root)
        validation_dataset = GraphClassificationDataset(
            validation_frame, label_columns, args.graph_root
        )
        collate = collate_graph_samples
    else:
        train_dataset = ImageClassificationDataset(
            train_frame,
            label_columns,
            data_root=args.data_root,
            image_size=args.image_size,
            augment=True,
            seed=args.seed,
        )
        validation_dataset = ImageClassificationDataset(
            validation_frame,
            label_columns,
            data_root=args.data_root,
            image_size=args.image_size,
            augment=False,
            seed=args.seed,
        )
        collate = None
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        generator=generator,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        persistent_workers=args.workers > 0,
    )
    positives = train_frame[label_columns].sum(axis=0).to_numpy(dtype=np.float32)
    negatives = len(train_frame) - positives
    positive_weights = negatives / np.maximum(positives, 1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(
        args.model,
        len(PRIMARY_LABELS),
        image_size=args.image_size,
        node_dim=7,
        clinical_dim=9,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = args.output_dir / "best.pt"
    recovery_path = args.output_dir / "last.pt"
    recovery_hash_path = args.output_dir / "last.sha256"
    progress_path = args.output_dir / "history_progress.csv"
    history = []
    best_auroc = -np.inf
    stale_epochs = 0
    start_epoch = 1
    resume_count = 0
    signature = {
        "model": args.model,
        "labels": PRIMARY_LABELS,
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "val_manifest_sha256": sha256_file(args.val_manifest),
        "train_cases": len(train_dataset),
        "validation_cases": len(validation_dataset),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "seed": args.seed,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "amp": not args.no_amp,
    }
    if args.resume:
        if not recovery_path.is_file():
            raise FileNotFoundError(
                f"No epoch recovery checkpoint exists: {recovery_path}"
            )
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if recovery.get("format_version") != 1:
            raise RuntimeError("Unsupported recovery checkpoint format")
        if recovery.get("test_evaluated") is not False:
            raise RuntimeError("Recovery checkpoint is not test-blind")
        if recovery.get("signature") != signature:
            raise RuntimeError(
                "Resume configuration or cohort hashes do not match the saved run"
            )
        expected_best_hash = recovery.get("best_checkpoint_sha256")
        if expected_best_hash is not None:
            if not checkpoint_path.is_file():
                raise FileNotFoundError("The saved best.pt required for resume is missing")
            if sha256_file(checkpoint_path) != expected_best_hash:
                raise RuntimeError("best.pt changed after the last completed epoch")
        model.load_state_dict(recovery["model_state"])
        optimizer.load_state_dict(recovery["optimizer_state"])
        optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(recovery["scheduler_state"])
        generator.set_state(recovery["data_loader_generator_state"].cpu())
        restore_rng_state(recovery["rng_state"])
        history = list(recovery["history"])
        best_auroc = float(recovery["best_auroc"])
        stale_epochs = int(recovery["stale_epochs"])
        start_epoch = int(recovery["epoch_completed"]) + 1
        resume_count = int(recovery.get("resume_count", 0)) + 1
        print(
            json.dumps(
                {
                    "resume": True,
                    "completed_epochs": start_epoch - 1,
                    "next_epoch": start_epoch,
                    "resume_count": resume_count,
                    "test_cases_accessed": 0,
                },
                indent=2,
            )
        )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "model": args.model,
                "train_cases": len(train_dataset),
                "validation_cases": len(validation_dataset),
                "test_cases_accessed": 0,
                "labels": PRIMARY_LABELS,
                "parameters": parameters,
                "device": str(device),
            },
            indent=2,
        )
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            amp=not args.no_amp,
        )
        validation_probabilities, validation_targets = predict(
            model, validation_loader, device
        )
        validation_metrics = multilabel_metrics(
            validation_probabilities, validation_targets, thresholds=0.5
        )
        macro_auroc = float(validation_metrics["macro"]["auroc"])
        scheduler.step(macro_auroc)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_macro_auroc": macro_auroc,
            "validation_macro_auprc": float(validation_metrics["macro"]["auprc"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row))
        if np.isfinite(macro_auroc) and macro_auroc > best_auroc:
            best_auroc = macro_auroc
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                model_name=args.model,
                label_names=PRIMARY_LABELS,
                epoch=epoch,
                validation_macro_auroc=macro_auroc,
                seed=args.seed,
            )
        else:
            stale_epochs += 1
        pd.DataFrame(history).to_csv(progress_path, index=False)
        best_checkpoint_hash = (
            sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
        )
        save_training_state(
            recovery_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader_generator=generator,
            epoch_completed=epoch,
            best_auroc=best_auroc,
            stale_epochs=stale_epochs,
            history=history,
            signature=signature,
            best_checkpoint_sha256=best_checkpoint_hash,
            resume_count=resume_count,
        )
        recovery_hash = sha256_file(recovery_path)
        recovery_hash_path.write_text(
            f"{recovery_hash}  last.pt\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "epoch_recovery_saved": epoch,
                    "recovery_checkpoint": str(recovery_path),
                    "recovery_sha256": recovery_hash,
                }
            )
        )
        if stale_epochs >= args.patience:
            break
    if not checkpoint_path.is_file():
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_probabilities, validation_targets = predict(model, validation_loader, device)
    thresholds = select_f1_thresholds(validation_probabilities, validation_targets)
    final_validation_metrics = multilabel_metrics(
        validation_probabilities, validation_targets, thresholds=thresholds
    )
    for name, metrics in zip(PRIMARY_LABELS, final_validation_metrics["per_label"]):
        metrics["label"] = name
    checkpoint["validation_thresholds"] = thresholds.tolist()
    checkpoint["validation_metrics"] = _json_safe(final_validation_metrics)
    checkpoint["positive_weights"] = positive_weights.tolist()
    checkpoint["test_evaluated"] = False
    torch.save(checkpoint, checkpoint_path)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    summary = {
        "artifact": "Objective 2 validation-selected classifier",
        "model": args.model,
        "parameters": parameters,
        "train_cases": len(train_dataset),
        "validation_cases": len(validation_dataset),
        "test_cases_accessed": 0,
        "test_evaluated": False,
        "labels": PRIMARY_LABELS,
        "best_epoch": int(checkpoint["epoch"]),
        "validation_thresholds": thresholds.tolist(),
        "validation_metrics": _json_safe(final_validation_metrics),
        "seed": args.seed,
        "resume_count": resume_count,
        "epoch_recovery_enabled": True,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with (args.output_dir / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    (args.output_dir / "best.sha256").write_text(
        f"{checkpoint_hash}  best.pt\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": int(checkpoint["epoch"]),
                "validation_macro_auroc": final_validation_metrics["macro"]["auroc"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "test_evaluated": False,
                "resume_count": resume_count,
            },
            indent=2,
        )
    )
    print("OBJECTIVE 2 MODEL TRAINING AND VALIDATION SUCCESSFUL")


if __name__ == "__main__":
    main()
