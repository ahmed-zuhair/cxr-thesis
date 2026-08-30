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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("image_only", "multimodal"), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
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
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    # Frozen DenseNet batch-normalisation statistics must not drift.
    model.image_encoder.eval()
    total_loss = 0.0
    correct = 0
    tokens = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        reports = batch["report_ids"].to(device, non_blocking=True)
        decoder_input = reports[:, :-1]
        target = reports[:, 1:]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(image, clinical, decoder_input)["report_logits"]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
        mask = target.ne(0)
        correct += int((logits.argmax(dim=-1).eq(target) & mask).sum().item())
        count = int(mask.sum().item())
        tokens += count
        total_loss += float(loss.item()) * count
    return total_loss / max(tokens, 1), correct / max(tokens, 1)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
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
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
    args.output_dir.mkdir(parents=True)
    vocabulary_path = args.output_dir / "vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(vocabulary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_checksum(vocabulary_path)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(
                model, validation_loader, criterion, device, None
            )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_token_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_perplexity": float(math.exp(min(val_loss, 20.0))),
            "validation_token_accuracy": val_accuracy,
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
        last_path = args.output_dir / "last.pt"
        atomic_torch_save(payload, last_path)
        write_checksum(last_path)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_path = args.output_dir / "best.pt"
            atomic_torch_save(payload, best_path)
            write_checksum(best_path)
        print(json.dumps(record, sort_keys=True))

    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
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
