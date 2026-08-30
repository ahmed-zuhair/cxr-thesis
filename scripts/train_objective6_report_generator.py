#!/usr/bin/env python3
"""Train a test-blind PadChest clinical report generator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from cxr_thesis.objective6.data import ReportGenerationDataset, collate_reports
from cxr_thesis.objective6.models import DenseNetTransformerReportGenerator
from cxr_thesis.objective6.text import ReportVocabulary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_checksum(path: Path) -> str:
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("image_only", "multimodal"), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--maximum-length", type=int, default=160)
    parser.add_argument("--maximum-vocabulary-size", type=int, default=12000)
    parser.add_argument("--minimum-token-frequency", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-cases", type=int, default=0)
    parser.add_argument("--validation-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    amp: bool,
    accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    # Frozen DenseNet batch-normalisation statistics must not drift.
    model.image_encoder.eval()
    total_loss = 0.0
    correct = 0
    tokens = 0
    if training:
        assert scaler is not None
        optimizer.zero_grad(set_to_none=True)
    batches = len(loader)
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        reports = batch["report_ids"].to(device, non_blocking=True)
        decoder_input = reports[:, :-1]
        target = reports[:, 1:]
        group_offset = batch_index % accumulation_steps
        group_size = min(accumulation_steps, batches - (batch_index - group_offset))
        with torch.set_grad_enabled(training), torch.amp.autocast(
            "cuda", enabled=amp and device.type == "cuda"
        ):
            logits = model(image, clinical, decoder_input)["report_logits"]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
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
        mask = target.ne(0)
        correct += int((logits.argmax(dim=-1).eq(target) & mask).sum().item())
        count = int(mask.sum().item())
        tokens += count
        total_loss += float(loss.item()) * count
    return total_loss / max(tokens, 1), correct / max(tokens, 1)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output exists and will not be overwritten: {args.output_dir}")
    for path in (args.train_manifest, args.val_manifest, args.source_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hash = sha256(args.source_checkpoint)
    if source_hash != args.expected_source_sha256:
        raise RuntimeError("Objective 5 PadChest encoder checkpoint hash changed")
    seed_everything(args.seed)
    train = pd.read_csv(args.train_manifest, low_memory=False)
    validation = pd.read_csv(args.val_manifest, low_memory=False)
    if args.train_cases:
        train = train.iloc[: args.train_cases].copy()
    if args.validation_cases:
        validation = validation.iloc[: args.validation_cases].copy()
    if set(train["patient_id"].astype(str)) & set(validation["patient_id"].astype(str)):
        raise RuntimeError("Training/validation patient leakage")

    vocabulary_path = args.output_dir / "vocabulary.json"
    if args.resume:
        if not vocabulary_path.is_file():
            raise FileNotFoundError(vocabulary_path)
        vocabulary = ReportVocabulary.from_dict(
            json.loads(vocabulary_path.read_text(encoding="utf-8"))
        )
    else:
        vocabulary = ReportVocabulary.build(
            train["report"],
            minimum_frequency=args.minimum_token_frequency,
            maximum_size=args.maximum_vocabulary_size,
        )
    train_dataset = ReportGenerationDataset(
        train, vocabulary, image_size=args.image_size, maximum_length=args.maximum_length
    )
    validation_dataset = ReportGenerationDataset(
        validation, vocabulary, image_size=args.image_size, maximum_length=args.maximum_length
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
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    model = DenseNetTransformerReportGenerator(
        len(vocabulary.tokens),
        maximum_length=args.maximum_length,
        pretrained=False,
        freeze_image_encoder=True,
        use_clinical=args.variant == "multimodal",
    )
    model.load_objective5_encoder(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=not args.no_amp and device.type == "cuda"
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
    if not args.resume:
        args.output_dir.mkdir(parents=True)
        vocabulary_path.write_text(
            json.dumps(vocabulary.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_checksum(vocabulary_path)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    resume_count = 0
    signature = {
        "variant": args.variant,
        "train_manifest_sha256": sha256(args.train_manifest),
        "validation_manifest_sha256": sha256(args.val_manifest),
        "source_checkpoint_sha256": source_hash,
        "training_cases": len(train),
        "validation_cases": len(validation),
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "image_size": args.image_size,
        "maximum_length": args.maximum_length,
        "vocabulary_size": len(vocabulary.tokens),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp": not args.no_amp,
    }
    if args.resume:
        recovery_path = args.output_dir / "last.pt"
        if not recovery_path.is_file():
            raise FileNotFoundError(recovery_path)
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if recovery.get("signature") != signature:
            raise RuntimeError("Resume configuration or protected inputs changed")
        if recovery.get("test_evaluated") is not False:
            raise RuntimeError("Recovery checkpoint is not test-blind")
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
        print(json.dumps({
            "resume": True,
            "completed_epochs": start_epoch - 1,
            "next_epoch": start_epoch,
            "resume_count": resume_count,
            "test_cases_accessed": 0,
        }, indent=2))
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer,
            amp=not args.no_amp, accumulation_steps=args.accumulation_steps,
            scaler=scaler,
        )
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                model, validation_loader, criterion, device, None,
                amp=not args.no_amp,
            )
        scheduler.step(val_loss)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_token_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_perplexity": float(math.exp(min(val_loss, 20.0))),
            "validation_token_accuracy": val_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        payload: dict[str, object] = {
            "format_version": 1,
            "model": "densenet_transformer_report_generator",
            "variant": args.variant,
            "model_state": model.state_dict(),
            "model_config": {
                "vocabulary_size": len(vocabulary.tokens),
                "maximum_length": args.maximum_length,
                "use_clinical": args.variant == "multimodal",
            },
            "vocabulary": vocabulary.to_dict(),
            "epoch": epoch,
            "history": history,
            "source_checkpoint_sha256": source_hash,
            "train_manifest_sha256": sha256(args.train_manifest),
            "validation_manifest_sha256": sha256(args.val_manifest),
            "training_cases": len(train),
            "validation_cases": len(validation),
            "test_cases_accessed": 0,
            "test_evaluated": False,
        }
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_path = args.output_dir / "best.pt"
            atomic_torch_save(payload, best_path)
            write_checksum(best_path)
        else:
            stale_epochs += 1
        recovery_payload = {
            **payload,
            "epoch_completed": epoch,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "data_loader_generator_state": generator.get_state(),
            "rng_state": capture_rng_state(),
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "resume_count": resume_count,
            "signature": signature,
        }
        last_path = args.output_dir / "last.pt"
        atomic_torch_save(recovery_payload, last_path)
        write_checksum(last_path)
        progress_path = args.output_dir / "history_progress.csv"
        with progress_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(record, sort_keys=True))
        if stale_epochs >= args.patience:
            print(json.dumps({"early_stopping_epoch": epoch, "best_epoch": best_epoch}))
            break

    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best_checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state"])
    model.eval()
    smoke_batch = next(iter(validation_loader))
    with torch.no_grad():
        generated = model.generate(
            smoke_batch["image"].to(device),
            smoke_batch["clinical"].to(device),
            maximum_length=min(args.maximum_length, 64),
        ).cpu()
    generated_content = generated[:, 1:].ne(0) & generated[:, 1:].ne(2)
    generated_nonempty_fraction = float(generated_content.any(dim=1).float().mean())
    validation_summary = {
        "artifact": "Objective 6 validation-only report-generator result",
        "research_result": not (len(train) < 29283 or len(validation) < 6280),
        "variant": args.variant,
        "training_cases": len(train),
        "validation_cases": len(validation),
        "vocabulary_size": len(vocabulary.tokens),
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "validation_perplexity": float(math.exp(min(best_loss, 20.0))),
        "real_data_generation_cases": int(generated.shape[0]),
        "generated_nonempty_fraction": generated_nonempty_fraction,
        "test_manifest_opened": False,
        "test_cases_accessed": 0,
        "test_evaluated": False,
        "source_checkpoint_sha256": source_hash,
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "epochs_completed": int(history[-1]["epoch"]),
        "resume_count": resume_count,
        "private_recovery_ready": True,
    }
    summary_path = args.output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(validation_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation_summary, indent=2, sort_keys=True))
    print("OBJECTIVE 6 TEST-BLIND REPORT-GENERATOR TRAINING SUCCESSFUL")


if __name__ == "__main__":
    main()
