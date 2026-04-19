"""
train.py
========
Training loop for multi-label CXR classification on Kaggle.

Designed for Kaggle constraints:
- Mixed precision (autocast + GradScaler) — 2x speedup, half the memory
- Checkpoint after every epoch — survives 9-hour session limit
- Resume from checkpoint — pick up where we left off in next session
- Per-epoch validation AUC logging — track convergence honestly
- Save best checkpoint (highest mean val AUC)

Usage on Kaggle (in a notebook cell):
    !cd /kaggle/working/cxr-thesis && python train.py \\
        --data_root /kaggle/input/datasets/organizations/nih-chest-xrays/data \\
        --model densenet121 \\
        --epochs 10 \\
        --batch_size 32 \\
        --lr 1e-4
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from nih_dataset import make_dataloaders, DISEASE_LABELS, NUM_CLASSES
from models import DenseNet121MultiLabel, count_parameters


# ---- Optional: HuggingFace upload of best checkpoint ------------------------
# Set HF_TOKEN env var + HF_REPO_ID arg to enable. Falls back gracefully if not.
def upload_to_hf(local_path: str, repo_id: str, token: str,
                 path_in_repo: str = "best.pth"):
    """Upload a file to a HuggingFace model repo. Silent failure on error."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[hf] uploaded {local_path} → {repo_id}/{path_in_repo}")
    except Exception as e:
        print(f"[hf] upload FAILED: {e}")
        print(f"[hf] checkpoint still saved locally at {local_path}")
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="Root containing Data_Entry_2017.csv and images_001..012/")
    p.add_argument("--model", type=str, default="densenet121",
                   choices=["densenet121"])  # more added later
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--checkpoint_dir", type=str, default="/kaggle/working/checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf_repo_id", type=str, default=None,
                   help="HF repo to push best.pth, e.g. 'username/cxr-thesis-checkpoints'")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> nn.Module:
    if name == "densenet121":
        return DenseNet121MultiLabel(num_classes=NUM_CLASSES, pretrained=True)
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run validation. Returns mean loss + per-class AUC array + mean AUC."""
    model.eval()
    losses = []
    all_logits, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, labels)
        losses.append(loss.item())
        all_logits.append(torch.sigmoid(logits).float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    # Per-class AUC (skip classes with no positives in this split — shouldn't happen on val)
    aucs = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        if all_labels[:, c].sum() > 0 and all_labels[:, c].sum() < len(all_labels):
            aucs[c] = roc_auc_score(all_labels[:, c], all_logits[:, c])
        else:
            aucs[c] = float("nan")
    mean_auc = np.nanmean(aucs)
    return float(np.mean(losses)), aucs, float(mean_auc)


def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}")

    # Data
    print("[data] building dataloaders...")
    train_loader, val_loader, test_loader, pos_weights = make_dataloaders(
        csv_path=os.path.join(args.data_root, "Data_Entry_2017.csv"),
        images_dir=args.data_root,
        train_val_list=os.path.join(args.data_root, "train_val_list.txt"),
        test_list=os.path.join(args.data_root, "test_list.txt"),
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    print(f"[data] train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # Model
    print(f"[model] building {args.model}...")
    model = build_model(args.model).to(device)
    print(f"[model] {count_parameters(model):,} trainable parameters")

    # Loss + optimizer
    pos_weights = pos_weights.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda")

    # Resume?
    start_epoch = 0
    best_mean_auc = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"[resume] loading {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_mean_auc = ckpt.get("best_mean_auc", 0.0)
        print(f"[resume] resuming at epoch {start_epoch}, best AUC so far: {best_mean_auc:.4f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(train_losses[-50:]):.4f}")

        scheduler.step()
        train_loss = float(np.mean(train_losses))

        # Validate
        val_loss, val_aucs, val_mean_auc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        print(f"\n[epoch {epoch+1}] "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_mean_auc={val_mean_auc:.4f}  "
              f"time={epoch_time/60:.1f}min")
        print("[per-class val AUC]")
        for name, auc in zip(DISEASE_LABELS, val_aucs):
            print(f"  {name:20s} {auc:.4f}")

        # Save checkpoint every epoch (Kaggle disconnect insurance)
        ckpt_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_mean_auc": best_mean_auc,
            "val_mean_auc": val_mean_auc,
            "val_aucs": val_aucs.tolist(),
            "args": vars(args),
        }
        save_checkpoint(ckpt_state, os.path.join(args.checkpoint_dir, "last.pth"))

        # Save best model separately
        if val_mean_auc > best_mean_auc:
            best_mean_auc = val_mean_auc
            ckpt_state["best_mean_auc"] = best_mean_auc
            best_path = os.path.join(args.checkpoint_dir, "best.pth")
            save_checkpoint(ckpt_state, best_path)
            print(f"[checkpoint] new best! mean_auc={best_mean_auc:.4f}")

            # Auto-upload to HF if configured (safest checkpoint location)
            hf_token = os.environ.get("HF_TOKEN")
            if args.hf_repo_id and hf_token:
                upload_to_hf(best_path, args.hf_repo_id, hf_token,
                             path_in_repo=f"{args.model}/best.pth")

    print(f"\n[done] best val mean AUC: {best_mean_auc:.4f}")
    print(f"[done] checkpoints in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
