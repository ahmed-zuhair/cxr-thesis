"""Train and validate the Objective 1 union-ROI U-Net."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.config import load_config
from cxr_thesis.objective1.manifest import validate_manifest
from cxr_thesis.objective1.segmentation import UNet2D, dice_score, hausdorff95, iou_score, segmentation_loss
from cxr_thesis.objective1.segmentation_data import ROISegmentationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train union lung/ROI segmentation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--config", default=str(REPOSITORY_ROOT / "configs" / "objective1" / "default.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        losses.append(float(segmentation_loss(logits, masks).item()))
        probabilities.extend(torch.sigmoid(logits)[:, 0].cpu().numpy())
        targets.extend(masks[:, 0].cpu().numpy())
    best_threshold, best_dice = 0.5, -1.0
    for threshold in np.linspace(0.30, 0.70, 9):
        score = float(np.mean([dice_score(prob >= threshold, target) for prob, target in zip(probabilities, targets)]))
        if score > best_dice:
            best_threshold, best_dice = float(threshold), score
    predictions = [prob >= best_threshold for prob in probabilities]
    hd95_values = np.asarray([hausdorff95(pred, target) for pred, target in zip(predictions, targets)])
    finite_hd95 = hd95_values[np.isfinite(hd95_values)]
    metrics = {
        "threshold": best_threshold,
        "dice": best_dice,
        "iou": float(np.mean([iou_score(pred, target) for pred, target in zip(predictions, targets)])),
        "hd95": float(finite_hd95.mean()) if finite_hd95.size else float("inf"),
        "failure_rate": float(np.isinf(hd95_values).mean()),
    }
    return float(np.mean(losses)), metrics


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str, "study_id": str, "image_id": str})
    validate_manifest(manifest, require_files=True, root=args.data_root)
    train_data = ROISegmentationDataset(manifest, args.data_root, config.preprocessing, split="train", augment=True)
    val_data = ROISegmentationDataset(manifest, args.data_root, config.preprocessing, split="val", augment=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dice, stale = -1.0, 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_losses: list[float] = []
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = segmentation_loss(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_losses.append(float(loss.item()))
        val_loss, metrics = validate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(training_losses)), "val_loss": val_loss, **metrics}
        history.append(row)
        print(json.dumps(row))
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": "UNet2D",
                    "channels": [32, 64, 128, 256],
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "objective1_config": str(Path(args.config).resolve()),
                    "seed": args.seed,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if stale >= args.patience:
            break

    checkpoint = output_dir / "best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (output_dir / "best.sha256").write_text(f"{digest}  best.pt\n", encoding="utf-8")
    print(json.dumps({"best_validation_dice": best_dice, "checkpoint": str(checkpoint), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()

