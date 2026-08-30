#!/usr/bin/env python3
"""Train the locked Objective 6 v1.1 clinical-guided report generator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from cxr_thesis.objective6.data import ReportGenerationDataset, collate_reports
from cxr_thesis.objective6.evaluation import parse_padchest6_labels
from cxr_thesis.objective6.models import DenseNetTransformerReportGenerator
from cxr_thesis.objective6.text import ReportVocabulary

PROTOCOL_SHA256 = "279e4fe83da6d82afcbcce595b5596980ca970fae958a16264d3b3e5172eb1a1"
LOCK_SHA256 = "b840440da16023c0169eb3f32c0f4ce7a20ecfa34f8f6b6bfa8ef20511aa53e6"
TRAIN_SHA256 = "278addf3c0a216bb206b4e4b79364f26bacbee977f3209e9275e2abbd8fda7d7"
VAL_SHA256 = "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
V1_CHECKPOINT_SHA256 = "18aa4293195b77aaf04df1ba310431df83f75e51ee6aa5837ff43a48a8ec10d3"
VARIANT = "clinical_guided_multimodal_v1_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--enhancement-protocol", type=Path, required=True)
    parser.add_argument("--enhancement-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--maximum-length", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clinical-loss-weight", type=float, default=0.35)
    parser.add_argument("--repetition-loss-weight", type=float, default=0.02)
    parser.add_argument("--train-cases", type=int, default=0)
    parser.add_argument("--validation-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
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


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def macro_f1(reference: np.ndarray, prediction: np.ndarray) -> float:
    values = []
    for column in range(reference.shape[1]):
        target = reference[:, column].astype(bool)
        estimate = prediction[:, column].astype(bool)
        true_positive = np.logical_and(target, estimate).sum()
        false_positive = np.logical_and(~target, estimate).sum()
        false_negative = np.logical_and(target, ~estimate).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(values))


def adjacent_repetition_penalty(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] < 2:
        return logits.new_zeros(())
    probability = logits.float().softmax(dim=-1)
    similarity = (probability[:, 1:] * probability[:, :-1]).sum(dim=-1)
    mask = target[:, 1:].ne(0) & target[:, :-1].ne(0)
    mask &= target[:, 1:].ne(target[:, :-1])
    if not bool(mask.any()):
        return logits.new_zeros(())
    return similarity[mask].mean().to(logits.dtype)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    report_criterion: nn.Module,
    clinical_criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    clinical_weight: float,
    repetition_weight: float,
    amp: bool,
    accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    model.image_encoder.eval()
    totals = {"total": 0.0, "report": 0.0, "clinical": 0.0, "repetition": 0.0}
    tokens = 0
    clinical_reference = []
    clinical_prediction = []
    if training:
        assert scaler is not None
        optimizer.zero_grad(set_to_none=True)
    batches = len(loader)
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        labels = batch["clinical_labels"].to(device, non_blocking=True)
        reports = batch["report_ids"].to(device, non_blocking=True)
        decoder_input = reports[:, :-1]
        target = reports[:, 1:]
        group_offset = batch_index % accumulation_steps
        group_size = min(accumulation_steps, batches - (batch_index - group_offset))
        with torch.set_grad_enabled(training), torch.amp.autocast(
            "cuda", enabled=amp and device.type == "cuda"
        ):
            output = model(image, clinical, decoder_input)
            logits = output["report_logits"]
            clinical_logits = output["clinical_logits"]
            report_loss = report_criterion(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
            )
            clinical_loss = clinical_criterion(clinical_logits, labels)
            repetition_loss = adjacent_repetition_penalty(logits, target)
            loss = (
                report_loss
                + clinical_weight * clinical_loss
                + repetition_weight * repetition_loss
            )
            if training:
                scaler.scale(loss / group_size).backward()
        if training and group_offset + 1 == group_size:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        count = int(target.ne(0).sum().item())
        tokens += count
        totals["total"] += float(loss.item()) * count
        totals["report"] += float(report_loss.item()) * count
        totals["clinical"] += float(clinical_loss.item()) * count
        totals["repetition"] += float(repetition_loss.item()) * count
        clinical_reference.append(labels.detach().cpu().numpy())
        clinical_prediction.append(clinical_logits.detach().gt(0).cpu().numpy())
    reference = np.concatenate(clinical_reference)
    prediction = np.concatenate(clinical_prediction)
    return {
        "total_loss": totals["total"] / max(tokens, 1),
        "report_loss": totals["report"] / max(tokens, 1),
        "clinical_loss": totals["clinical"] / max(tokens, 1),
        "repetition_loss": totals["repetition"] / max(tokens, 1),
        "auxiliary_macro_f1": macro_f1(reference, prediction),
    }


def main() -> None:
    args = parse_args()
    locked_values = {
        "patience": 4, "batch_size": 8,
        "accumulation_steps": 2, "image_size": 320, "maximum_length": 160,
        "learning_rate": 1e-4, "encoder_learning_rate": 2e-5,
        "weight_decay": 1e-4, "clinical_loss_weight": 0.35,
        "repetition_loss_weight": 0.02, "seed": 42,
    }
    for name, value in locked_values.items():
        if getattr(args, name) != value:
            raise RuntimeError(f"Locked Objective 6 v1.1 option changed: {name}")
    smoke = bool(args.train_cases or args.validation_cases)
    if smoke:
        if (
            args.epochs != 1
            or args.train_cases != 1024
            or args.validation_cases != 512
        ):
            raise RuntimeError("Objective 6 v1.1 smoke configuration changed")
    elif args.epochs != 12:
        raise RuntimeError("Locked Objective 6 v1.1 full-training epochs changed")
    protected = {
        args.train_manifest: TRAIN_SHA256,
        args.val_manifest: VAL_SHA256,
        args.v1_checkpoint: V1_CHECKPOINT_SHA256,
        args.enhancement_protocol: PROTOCOL_SHA256,
        args.enhancement_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 v1.1 input changed: {path}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    seed_everything(args.seed)
    train = pd.read_csv(args.train_manifest, low_memory=False)
    validation = pd.read_csv(args.val_manifest, low_memory=False)
    if args.train_cases:
        train = train.iloc[: args.train_cases].copy()
    if args.validation_cases:
        validation = validation.iloc[: args.validation_cases].copy()
    if set(train["patient_id"].astype(str)) & set(validation["patient_id"].astype(str)):
        raise RuntimeError("Objective 6 v1.1 patient leakage")
    v1 = torch.load(args.v1_checkpoint, map_location="cpu", weights_only=False)
    if (
        v1.get("variant") != "multimodal"
        or v1.get("test_evaluated") is not False
        or v1.get("training_cases") != 29283
        or v1.get("validation_cases") != 6280
    ):
        raise RuntimeError("Objective 6 v1 checkpoint metadata changed")
    vocabulary = ReportVocabulary.from_dict(v1["vocabulary"])
    model = DenseNetTransformerReportGenerator(
        len(vocabulary.tokens), maximum_length=args.maximum_length,
        pretrained=False, freeze_image_encoder=True, use_clinical=True,
        use_concept_token=True,
    )
    load = model.load_state_dict(v1["model_state"], strict=False)
    expected_missing = {
        "concept_projection.0.weight", "concept_projection.0.bias",
        "concept_projection.2.weight", "concept_projection.2.bias",
    }
    if set(load.missing_keys) != expected_missing or load.unexpected_keys:
        raise RuntimeError(
            f"Objective 6 v1.1 initialization mismatch: {load}"
        )
    model.set_final_image_block_trainable()
    trainable_encoder = [
        parameter for parameter in model.image_encoder.parameters()
        if parameter.requires_grad
    ]
    trainable_other = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("image_encoder.")
    ]
    if not trainable_encoder or not trainable_other:
        raise RuntimeError("Objective 6 v1.1 trainable parameter groups are empty")

    train_dataset = ReportGenerationDataset(
        train, vocabulary, image_size=args.image_size,
        maximum_length=args.maximum_length, include_clinical_labels=True,
    )
    validation_dataset = ReportGenerationDataset(
        validation, vocabulary, image_size=args.image_size,
        maximum_length=args.maximum_length, include_clinical_labels=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, generator=generator,
        collate_fn=collate_reports,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_reports,
    )
    training_targets = np.stack(
        train["labels"].map(parse_padchest6_labels).to_numpy()
    ).astype(np.float32)
    positives = training_targets.sum(axis=0)
    negatives = len(training_targets) - positives
    if (positives <= 0).any():
        raise RuntimeError("Objective 6 v1.1 training target is degenerate")
    positive_weight = torch.tensor(negatives / positives, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_other, "lr": args.learning_rate},
            {"params": trainable_encoder, "lr": args.encoder_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=not args.no_amp and device.type == "cuda"
    )
    report_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
    clinical_criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight.to(device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary_path = args.output_dir / "vocabulary.json"
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    resume_count = 0
    signature = {
        "variant": VARIANT,
        "protocol_sha256": PROTOCOL_SHA256,
        "lock_sha256": LOCK_SHA256,
        "train_manifest_sha256": TRAIN_SHA256,
        "validation_manifest_sha256": VAL_SHA256,
        "v1_checkpoint_sha256": V1_CHECKPOINT_SHA256,
        "training_cases": len(train), "validation_cases": len(validation),
        "epochs": args.epochs, **locked_values,
        "amp": not args.no_amp,
    }
    if args.resume:
        recovery_path = args.output_dir / "last.pt"
        if not recovery_path.is_file():
            raise FileNotFoundError(recovery_path)
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if recovery.get("signature") != signature:
            raise RuntimeError("Objective 6 v1.1 resume signature changed")
        if recovery.get("test_evaluated") is not False:
            raise RuntimeError("Objective 6 v1.1 recovery is not test-blind")
        model.load_state_dict(recovery["model_state"])
        optimizer.load_state_dict(recovery["optimizer_state"])
        scheduler.load_state_dict(recovery["scheduler_state"])
        scaler.load_state_dict(recovery["scaler_state"])
        generator.set_state(recovery["data_loader_generator_state"].cpu())
        restore_rng_state(recovery["rng_state"])
        history = list(recovery["history"])
        best_loss = float(recovery["best_loss"])
        best_epoch = int(recovery["best_epoch"])
        stale_epochs = int(recovery["stale_epochs"])
        start_epoch = int(recovery["epoch_completed"]) + 1
        resume_count = int(recovery.get("resume_count", 0)) + 1
    else:
        vocabulary_path.write_text(
            json.dumps(vocabulary.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_checksum(vocabulary_path)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, report_criterion, clinical_criterion, device,
            optimizer, clinical_weight=args.clinical_loss_weight,
            repetition_weight=args.repetition_loss_weight,
            amp=not args.no_amp, accumulation_steps=args.accumulation_steps,
            scaler=scaler,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, validation_loader, report_criterion, clinical_criterion,
                device, None, clinical_weight=args.clinical_loss_weight,
                repetition_weight=args.repetition_loss_weight,
                amp=not args.no_amp,
            )
        scheduler.step(validation_metrics["total_loss"])
        record: dict[str, float | int] = {"epoch": epoch}
        record.update({f"train_{key}": value for key, value in train_metrics.items()})
        record.update({
            f"validation_{key}": value for key, value in validation_metrics.items()
        })
        record["decoder_learning_rate"] = float(optimizer.param_groups[0]["lr"])
        record["encoder_learning_rate"] = float(optimizer.param_groups[1]["lr"])
        history.append(record)
        payload: dict[str, object] = {
            "format_version": 2, "model": "densenet_transformer_report_generator",
            "variant": VARIANT, "model_state": model.state_dict(),
            "model_config": {
                "vocabulary_size": len(vocabulary.tokens),
                "maximum_length": args.maximum_length,
                "use_clinical": True, "use_concept_token": True,
                "beam_width": 3, "length_normalization_alpha": 0.7,
                "no_repeat_ngram_size": 4,
            },
            "vocabulary": vocabulary.to_dict(), "epoch": epoch,
            "history": history, "protocol_sha256": PROTOCOL_SHA256,
            "lock_sha256": LOCK_SHA256,
            "v1_checkpoint_sha256": V1_CHECKPOINT_SHA256,
            "train_manifest_sha256": TRAIN_SHA256,
            "validation_manifest_sha256": VAL_SHA256,
            "training_cases": len(train), "validation_cases": len(validation),
            "clinical_positive_weights": positive_weight.tolist(),
            "test_cases_accessed": 0, "test_evaluated": False,
        }
        if validation_metrics["total_loss"] < best_loss:
            best_loss = validation_metrics["total_loss"]
            best_epoch = epoch
            stale_epochs = 0
            best_path = args.output_dir / "best.pt"
            atomic_torch_save(payload, best_path)
            write_checksum(best_path)
        else:
            stale_epochs += 1
        recovery = {
            **payload, "epoch_completed": epoch,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "data_loader_generator_state": generator.get_state(),
            "rng_state": capture_rng_state(), "best_loss": best_loss,
            "best_epoch": best_epoch, "stale_epochs": stale_epochs,
            "resume_count": resume_count, "signature": signature,
        }
        last_path = args.output_dir / "last.pt"
        atomic_torch_save(recovery, last_path)
        write_checksum(last_path)
        with (args.output_dir / "history_progress.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(record, sort_keys=True), flush=True)
        if stale_epochs >= args.patience:
            print(json.dumps({"early_stopping_epoch": epoch, "best_epoch": best_epoch}))
            break

    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    model.eval()
    smoke_batch = next(iter(validation_loader))
    with torch.no_grad():
        generated = model.generate_beam(
            smoke_batch["image"].to(device), smoke_batch["clinical"].to(device),
            maximum_length=min(args.maximum_length, 64), beam_width=3,
            length_normalization_alpha=0.7, no_repeat_ngram_size=4,
        ).cpu()
    generated_content = generated[:, 1:].ne(0) & generated[:, 1:].ne(2)
    summary = {
        "artifact": "Objective 6 v1.1 validation-only enhancement result",
        "research_result": not (len(train) < 29283 or len(validation) < 6280),
        "variant": VARIANT, "training_cases": len(train),
        "validation_cases": len(validation), "best_epoch": best_epoch,
        "epochs_completed": int(history[-1]["epoch"]),
        "validation_total_joint_loss": best_loss,
        "validation_report_perplexity": float(math.exp(min(
            float(history[best_epoch - 1]["validation_report_loss"]), 20.0
        ))),
        "validation_auxiliary_macro_f1": float(
            history[best_epoch - 1]["validation_auxiliary_macro_f1"]
        ),
        "generated_nonempty_fraction": float(
            generated_content.any(dim=1).float().mean()
        ),
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "protocol_sha256": PROTOCOL_SHA256, "lock_sha256": LOCK_SHA256,
        "v1_checkpoint_sha256": V1_CHECKPOINT_SHA256,
        "resume_count": resume_count, "private_recovery_ready": True,
        "test_manifest_opened": False, "test_cases_accessed": 0,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("OBJECTIVE 6 V1.1 TEST-BLIND ENHANCEMENT TRAINING SUCCESSFUL")


if __name__ == "__main__":
    main()
