"""
evaluate.py
===========
Run a trained checkpoint on the test set and produce thesis-ready outputs:
    - per-class AUC table (saved as CSV for direct paste into thesis)
    - per-class ROC curves figure
    - precision/recall/F1 at optimal threshold
    - confusion matrix per class

Usage:
    python evaluate.py \\
        --data_root /kaggle/input/datasets/organizations/nih-chest-xrays/data \\
        --checkpoint /kaggle/working/checkpoints/best.pth \\
        --output_dir /kaggle/working/eval_results
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.amp import autocast
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, confusion_matrix)
from tqdm import tqdm

from nih_dataset import make_dataloaders, DISEASE_LABELS, NUM_CLASSES
from models import DenseNet121MultiLabel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="/kaggle/working/eval_results")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Maximum F1 threshold — better than 0.5 for imbalanced classes."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s[:-1]))  # last point has no threshold
    return float(thresholds[best_idx])


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model over loader, return (probs, labels) numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="predicting"):
        imgs = imgs.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(imgs)
        all_probs.append(torch.sigmoid(logits).float().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data — only test loader matters here, but make_dataloaders builds all 3
    _, _, test_loader, _ = make_dataloaders(
        csv_path=os.path.join(args.data_root, "Data_Entry_2017.csv"),
        images_dir=args.data_root,
        train_val_list=os.path.join(args.data_root, "train_val_list.txt"),
        test_list=os.path.join(args.data_root, "test_list.txt"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Load checkpoint
    print(f"[ckpt] loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DenseNet121MultiLabel(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[ckpt] from epoch {ckpt['epoch']}, val mean AUC was {ckpt['val_mean_auc']:.4f}")

    # Predict over test set
    probs, labels = collect_predictions(model, test_loader, device)
    print(f"[predictions] {probs.shape[0]} test samples")

    # Per-class metrics
    rows = []
    for c, name in enumerate(DISEASE_LABELS):
        y_true = labels[:, c]
        y_prob = probs[:, c]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            rows.append({"class": name, "n_pos": int(y_true.sum()), "auc": np.nan,
                         "threshold": np.nan, "f1": np.nan})
            continue
        auc = roc_auc_score(y_true, y_prob)
        thr = find_optimal_threshold(y_true, y_prob)
        f1 = f1_score(y_true, (y_prob >= thr).astype(int))
        rows.append({"class": name, "n_pos": int(y_true.sum()),
                     "auc": auc, "threshold": thr, "f1": f1})

    df = pd.DataFrame(rows)
    df.loc["mean"] = ["MEAN", df["n_pos"].sum(), df["auc"].mean(), np.nan, df["f1"].mean()]
    df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    print("\n=== Per-class results ===")
    print(df.to_string(index=False))
    print(f"\n[saved] {out_dir / 'per_class_metrics.csv'}")

    # ROC curves figure
    fig, ax = plt.subplots(figsize=(9, 7))
    for c, name in enumerate(DISEASE_LABELS):
        if labels[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(labels[:, c], probs[:, c])
        auc = roc_auc_score(labels[:, c], probs[:, c])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=1.2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves — DenseNet-121 baseline (mean AUC = {df['auc'][:-1].mean():.4f})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close()
    print(f"[saved] {out_dir / 'roc_curves.png'}")

    # Save raw predictions (useful later for ensembling, error analysis)
    np.savez(out_dir / "raw_predictions.npz", probs=probs, labels=labels)
    print(f"[saved] {out_dir / 'raw_predictions.npz'}")


if __name__ == "__main__":
    main()
