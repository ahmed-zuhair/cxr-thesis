#!/usr/bin/env python3
"""Train one paired Objective 3 head using frozen private GAT embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective2.metrics import (
    multilabel_metrics,
    select_f1_thresholds,
)
from cxr_thesis.objective2.training import (
    optimizer_state_to_device,
    restore_rng_state,
    save_training_state,
    seed_everything,
)
from cxr_thesis.objective3.models import (
    HybridGraphHead,
    bottleneck_parameter_count,
)
from cxr_thesis.objective3.training import (
    apply_standardizer,
    fit_standardizer,
    initialize_shared_layers,
    labels_from_manifest,
    load_embedding_corpus,
    make_loader,
    positive_weights,
    predict,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one test-blind paired Objective 3 bottleneck head"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=("quantum", "classical_matched"),
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-gat-sha256", required=True)
    parser.add_argument("--expected-train-cases", type=int, default=30_000)
    parser.add_argument("--expected-val-cases", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_manifest(
    path: Path,
    split: str,
    expected_hash: str,
    expected_cases: int,
) -> pd.DataFrame:
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RuntimeError(f"{split} manifest SHA-256 does not match")
    frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str})
    if len(frame) != expected_cases:
        raise RuntimeError(
            f"{split} manifest has {len(frame)} cases, expected {expected_cases}"
        )
    if set(frame["split"].astype(str).str.lower()) != {split}:
        raise RuntimeError(f"{split} manifest contains another split")
    return frame


def validate_embedding_index(
    index: dict[str, object], args: argparse.Namespace
) -> str:
    checks = {
        "encoder": index.get("encoder") == "gat",
        "encoder_frozen": index.get("encoder_frozen") is True,
        "embedding_dimension": index.get("embedding_dimension") == 160,
        "training_hash": index.get("train_manifest_sha256")
        == args.expected_train_sha256,
        "validation_hash": index.get("validation_manifest_sha256")
        == args.expected_val_sha256,
        "gat_hash": index.get("gat_checkpoint_sha256")
        == args.expected_gat_sha256,
        "training_cases": index.get("train_cases") == args.expected_train_cases,
        "validation_cases": index.get("validation_cases")
        == args.expected_val_cases,
        "test_manifest_not_opened": index.get("test_manifest_opened") is False,
        "test_labels_not_accessed": index.get("test_labels_accessed") is False,
        "test_not_evaluated": index.get("test_evaluated") is False,
        "private_only": index.get("allowed_for_public_upload") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Embedding recovery index failed checks: {failed}")
    stable = {
        "train_manifest_sha256": index["train_manifest_sha256"],
        "validation_manifest_sha256": index["validation_manifest_sha256"],
        "gat_checkpoint_sha256": index["gat_checkpoint_sha256"],
        "embedding_dimension": index["embedding_dimension"],
        "shards": [
            {
                key: record[key]
                for key in ("shard", "start", "stop", "cases", "split", "sha256")
            }
            for record in index["shards"]
        ],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_payload(
    *,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    macro_auroc: float,
    signature: dict[str, object],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "artifact": "Objective 3 validation-selected hybrid GAT head",
        "model_name": f"hybrid_gat_{args.variant}",
        "variant": args.variant,
        "model_state": model.state_dict(),
        "label_names": PRIMARY_LABELS,
        "epoch": int(epoch),
        "validation_macro_auroc": float(macro_auroc),
        "seed": int(args.seed),
        "embedding_dimension": 160,
        "projection_dimension": 4,
        "bottleneck_parameters": bottleneck_parameter_count(model.bottleneck),
        "total_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "standardizer_mean": scaler_mean.tolist(),
        "standardizer_standard_deviation": scaler_std.tolist(),
        "positive_weights": weights.tolist(),
        "training_signature": signature,
        "test_cases_accessed": 0,
        "test_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch-size must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {args.output_dir}")
    if args.resume and (args.output_dir / "validation_summary.json").is_file():
        raise RuntimeError("Training is already complete")

    train_frame = validate_manifest(
        args.train_manifest,
        "train",
        args.expected_train_sha256,
        args.expected_train_cases,
    )
    validation_frame = validate_manifest(
        args.val_manifest,
        "val",
        args.expected_val_sha256,
        args.expected_val_cases,
    )
    if set(train_frame["patient_id"]) & set(validation_frame["patient_id"]):
        raise RuntimeError("Patient leakage exists between training and validation")
    index_path = args.embedding_root / "private" / "embedding_recovery_index.json"
    shard_root = args.embedding_root / "private" / "shards"
    if not index_path.is_file() or not shard_root.is_dir():
        raise FileNotFoundError("The verified private embedding corpus is incomplete")
    recovery_index = json.loads(index_path.read_text(encoding="utf-8"))
    corpus_signature = validate_embedding_index(recovery_index, args)
    train_embeddings, validation_embeddings = load_embedding_corpus(
        train_frame, validation_frame, recovery_index, shard_root
    )

    if args.limit_train is not None:
        if not 0 < args.limit_train <= len(train_frame):
            raise ValueError("limit-train is outside the training cohort")
        train_frame = train_frame.iloc[: args.limit_train].copy()
        train_embeddings = train_embeddings[: args.limit_train]
    if args.limit_val is not None:
        if not 0 < args.limit_val <= len(validation_frame):
            raise ValueError("limit-val is outside the validation cohort")
        validation_frame = validation_frame.iloc[: args.limit_val].copy()
        validation_embeddings = validation_embeddings[: args.limit_val]

    train_labels = labels_from_manifest(train_frame, PRIMARY_LABELS)
    validation_labels = labels_from_manifest(validation_frame, PRIMARY_LABELS)
    scaler_mean, scaler_std = fit_standardizer(train_embeddings)
    train_embeddings = apply_standardizer(
        train_embeddings, scaler_mean, scaler_std
    )
    validation_embeddings = apply_standardizer(
        validation_embeddings, scaler_mean, scaler_std
    )
    weights = positive_weights(train_labels)

    seed_everything(args.seed)
    model = HybridGraphHead(
        len(PRIMARY_LABELS),
        bottleneck=args.variant,
        dropout=args.dropout,
    )
    initialize_shared_layers(model, args.seed)
    seed_everything(args.seed)
    device = torch.device("cpu")
    model.to(device)
    train_loader, generator = make_loader(
        train_embeddings,
        train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader, _ = make_loader(
        validation_embeddings,
        validation_labels,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(weights).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    progress_path = args.output_dir / "history_progress.csv"
    signature = {
        "objective": 3,
        "variant": args.variant,
        "labels": PRIMARY_LABELS,
        "train_manifest_sha256": args.expected_train_sha256,
        "validation_manifest_sha256": args.expected_val_sha256,
        "gat_checkpoint_sha256": args.expected_gat_sha256,
        "embedding_corpus_signature": corpus_signature,
        "train_cases": len(train_frame),
        "validation_cases": len(validation_frame),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "device": "cpu",
        "loss": "weighted_binary_cross_entropy",
        "normalization": "training_only_standardization",
    }
    history: list[dict[str, object]] = []
    best_auroc = -np.inf
    stale_epochs = 0
    start_epoch = 1
    resume_count = 0
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Recovery checkpoint is missing: {last_path}")
        recovery = torch.load(last_path, map_location="cpu", weights_only=False)
        if recovery.get("format_version") != 1:
            raise RuntimeError("Unsupported recovery format")
        if recovery.get("test_evaluated") is not False:
            raise RuntimeError("Recovery is not test-blind")
        if recovery.get("signature") != signature:
            raise RuntimeError("Recovery signature does not match this paired run")
        expected_best_hash = recovery.get("best_checkpoint_sha256")
        if expected_best_hash is not None and (
            not best_path.is_file() or sha256_file(best_path) != expected_best_hash
        ):
            raise RuntimeError("Recovered best.pt failed integrity verification")
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

    bottleneck_parameters = bottleneck_parameter_count(model.bottleneck)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if bottleneck_parameters != 24 or total_parameters != 2648:
        raise RuntimeError("Objective 3 parameter budget changed")
    print(
        json.dumps(
            {
                "objective": 3,
                "variant": args.variant,
                "train_cases": len(train_frame),
                "validation_cases": len(validation_frame),
                "test_cases_accessed": 0,
                "bottleneck_parameters": bottleneck_parameters,
                "total_trainable_parameters": total_parameters,
                "embedding_dimension": 160,
                "device": str(device),
                "paired_seed": args.seed,
                "resume_count": resume_count,
            },
            indent=2,
        ),
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        probabilities, targets = predict(model, validation_loader, device)
        metrics = multilabel_metrics(probabilities, targets, thresholds=0.5)
        macro_auroc = float(metrics["macro"]["auroc"])
        scheduler.step(macro_auroc)
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "validation_macro_auroc": macro_auroc,
            "validation_macro_auprc": float(metrics["macro"]["auprc"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if np.isfinite(macro_auroc) and macro_auroc > best_auroc:
            best_auroc = macro_auroc
            stale_epochs = 0
            atomic_torch_save(
                checkpoint_payload(
                    model=model,
                    args=args,
                    epoch=epoch,
                    macro_auroc=macro_auroc,
                    signature=signature,
                    scaler_mean=scaler_mean,
                    scaler_std=scaler_std,
                    weights=weights,
                ),
                best_path,
            )
        else:
            stale_epochs += 1
        pd.DataFrame(history).to_csv(progress_path, index=False)
        best_hash = sha256_file(best_path) if best_path.is_file() else None
        save_training_state(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader_generator=generator,
            epoch_completed=epoch,
            best_auroc=best_auroc,
            stale_epochs=stale_epochs,
            history=history,
            signature=signature,
            best_checkpoint_sha256=best_hash,
            resume_count=resume_count,
        )
        last_hash = sha256_file(last_path)
        (args.output_dir / "last.sha256").write_text(
            f"{last_hash}  last.pt\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "epoch_recovery_saved": epoch,
                    "recovery_sha256": last_hash,
                    "test_evaluated": False,
                }
            ),
            flush=True,
        )
        if stale_epochs >= args.patience:
            break

    if not best_path.is_file():
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    probabilities, targets = predict(model, validation_loader, device)
    thresholds = select_f1_thresholds(probabilities, targets)
    final_metrics = multilabel_metrics(
        probabilities, targets, thresholds=thresholds
    )
    for label, metrics in zip(PRIMARY_LABELS, final_metrics["per_label"]):
        metrics["label"] = label
    checkpoint["validation_thresholds"] = thresholds.tolist()
    checkpoint["validation_metrics"] = json_safe(final_metrics)
    checkpoint["test_evaluated"] = False
    atomic_torch_save(checkpoint, best_path)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    best_hash = sha256_file(best_path)
    (args.output_dir / "best.sha256").write_text(
        f"{best_hash}  best.pt\n", encoding="utf-8"
    )
    summary = {
        "artifact": "Objective 3 validation-selected paired hybrid GAT head",
        "objective": 3,
        "variant": args.variant,
        "model": f"hybrid_gat_{args.variant}",
        "seed": args.seed,
        "train_cases": len(train_frame),
        "validation_cases": len(validation_frame),
        "test_cases_accessed": 0,
        "test_evaluated": False,
        "embedding_dimension": 160,
        "bottleneck_parameters": bottleneck_parameters,
        "total_trainable_parameters": total_parameters,
        "best_epoch": int(checkpoint["epoch"]),
        "validation_thresholds": thresholds.tolist(),
        "validation_metrics": json_safe(final_metrics),
        "checkpoint_sha256": best_hash,
        "resume_count": resume_count,
        "epoch_recovery_enabled": True,
        "research_result": args.limit_train is None and args.limit_val is None,
        "training_configuration": signature,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pennylane": (
            __import__("pennylane").__version__
            if args.variant == "quantum"
            else None
        ),
    }
    atomic_json(summary, args.output_dir / "validation_summary.json")
    print(
        json.dumps(
            {
                "best_epoch": summary["best_epoch"],
                "validation_macro_auroc": final_metrics["macro"]["auroc"],
                "validation_macro_auprc": final_metrics["macro"]["auprc"],
                "checkpoint_sha256": best_hash,
                "test_evaluated": False,
                "research_result": summary["research_result"],
            },
            indent=2,
        )
    )
    print("OBJECTIVE 3 PAIRED HEAD TRAINING AND VALIDATION SUCCESSFUL")


if __name__ == "__main__":
    main()
