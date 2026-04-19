"""
train_gat_v2.py
===============
Training for improved GAT v2 (attention pooling + residuals).

Usage:
    python train_gat_v2.py \\
        --data_root /kaggle/input/.../data \\
        --backbone_ckpt /kaggle/working/.../best.pth \\
        --epochs 10 \\
        --batch_size 32 \\
        --lr 1e-4 \\
        --dropout 0.1 \\
        --hf_repo_id ahmed-zuhair/cxr-thesis-checkpoints
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
from gat_model_v2 import GATModelV2, count_trainable


def upload_to_hf(local_path, repo_id, token, path_in_repo):
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=local_path, path_in_repo=path_in_repo,
            repo_id=repo_id, repo_type="model",
        )
        print(f"[hf] uploaded {local_path} → {repo_id}/{path_in_repo}")
    except Exception as e:
        print(f"[hf] upload FAILED: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--backbone_ckpt", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--knn_k", type=int, default=3)
    p.add_argument("--no_knn", action="store_true")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", type=str, default="/kaggle/working/checkpoints_gat_v2")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--hf_repo_id", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stop_patience", type=int, default=4,
                   help="Stop if val AUC doesn't improve for N epochs")
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, all_probs, all_labels = [], [], []
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, labels)
        losses.append(loss.item())
        all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    aucs = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        if 0 < labels_arr[:, c].sum() < len(labels_arr):
            aucs[c] = roc_auc_score(labels_arr[:, c], probs[:, c])
        else:
            aucs[c] = float("nan")
    return float(np.mean(losses)), aucs, float(np.nanmean(aucs))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}")

    print("[data] building dataloaders...")
    train_loader, val_loader, test_loader, pos_weights = make_dataloaders(
        csv_path=os.path.join(args.data_root, "Data_Entry_2017.csv"),
        images_dir=args.data_root,
        train_val_list=os.path.join(args.data_root, "train_val_list.txt"),
        test_list=os.path.join(args.data_root, "test_list.txt"),
        batch_size=args.batch_size, image_size=args.image_size,
        num_workers=args.num_workers,
    )

    print(f"[model] GAT v2 with frozen DenseNet from {args.backbone_ckpt}")
    model = GATModelV2(
        pretrained_ckpt=args.backbone_ckpt,
        num_classes=NUM_CLASSES,
        knn_k=args.knn_k,
        use_knn=not args.no_knn,
        dropout=args.dropout,
    ).to(device)
    print(f"[model] trainable params: {count_trainable(model):,}")

    pos_weights = pos_weights.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda")

    start_epoch = 0
    best_mean_auc = 0.0
    epochs_since_improvement = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_mean_auc = ckpt.get("best_mean_auc", 0.0)

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
        val_loss, val_aucs, val_mean_auc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        print(f"\n[epoch {epoch+1}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"mean_auc={val_mean_auc:.4f}  time={epoch_time/60:.1f}min")
        for name, auc in zip(DISEASE_LABELS, val_aucs):
            print(f"  {name:20s} {auc:.4f}")

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
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_state, os.path.join(args.checkpoint_dir, "last.pth"))

        if val_mean_auc > best_mean_auc:
            best_mean_auc = val_mean_auc
            ckpt_state["best_mean_auc"] = best_mean_auc
            best_path = os.path.join(args.checkpoint_dir, "best.pth")
            torch.save(ckpt_state, best_path)
            print(f"[checkpoint] new best! mean_auc={best_mean_auc:.4f}")
            epochs_since_improvement = 0
            token = os.environ.get("HF_TOKEN")
            if args.hf_repo_id and token:
                upload_to_hf(best_path, args.hf_repo_id, token,
                             path_in_repo="gat_v2/best.pth")
        else:
            epochs_since_improvement += 1
            print(f"[patience] no improvement for {epochs_since_improvement}/{args.early_stop_patience} epochs")

        if epochs_since_improvement >= args.early_stop_patience:
            print(f"\n[early-stop] no improvement for {args.early_stop_patience} epochs, stopping")
            break

    print(f"\n[done] best val mean AUC: {best_mean_auc:.4f}")


if __name__ == "__main__":
    main()
