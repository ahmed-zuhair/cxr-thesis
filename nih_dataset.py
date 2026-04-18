"""
nih_dataset.py
==============
Data pipeline for NIH ChestX-ray14 multi-label classification.

Key design decisions (for thesis defense):
- Patient-wise splits: no patient appears in both train and test (prevents leakage)
- Official NIH train_val_list.txt / test_list.txt split used when available
- Multi-label binary vectors (14 classes) as targets
- CLAHE preprocessing applied on-the-fly (augments dataset without extra storage)
- Standard ImageNet normalization for transfer learning compatibility
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# -----------------------------------------------------------------------------
# The 14 disease labels in NIH ChestX-ray14, in fixed canonical order.
# Order matters: models will output a 14-dim vector aligned with this list.
# -----------------------------------------------------------------------------
DISEASE_LABELS: List[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(DISEASE_LABELS)

# ImageNet normalization stats — required for pretrained CNN compatibility
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# 1. LABEL PARSING
# =============================================================================
def parse_labels(finding_labels: str) -> np.ndarray:
    """
    Convert NIH label string like 'Effusion|Infiltration' to 14-dim binary vector.
    'No Finding' -> all zeros (valid: image has none of the 14 diseases).
    """
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    if finding_labels == "No Finding":
        return vec
    for label in finding_labels.split("|"):
        if label in DISEASE_LABELS:
            vec[DISEASE_LABELS.index(label)] = 1.0
    return vec


# =============================================================================
# 2. SPLIT BUILDER
# =============================================================================
def build_splits(
    csv_path: str,
    train_val_list: str,
    test_list: str,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build train/val/test DataFrames using official NIH patient-wise splits.

    Returns three DataFrames with columns:
        image_path, patient_id, labels (14-dim np.ndarray)
    """
    df = pd.read_csv(csv_path)
    # Rename columns for consistency
    df = df.rename(columns={
        "Image Index": "filename",
        "Finding Labels": "finding_labels",
        "Patient ID": "patient_id",
    })
    df["labels"] = df["finding_labels"].apply(parse_labels)

    # Load official split files (one filename per line)
    with open(train_val_list) as f:
        train_val_files = set(line.strip() for line in f if line.strip())
    with open(test_list) as f:
        test_files = set(line.strip() for line in f if line.strip())

    train_val_df = df[df["filename"].isin(train_val_files)].copy()
    test_df = df[df["filename"].isin(test_files)].copy()

    # Patient-wise train/val split (no patient in both)
    rng = np.random.default_rng(seed)
    patients = train_val_df["patient_id"].unique()
    rng.shuffle(patients)
    n_val = int(len(patients) * val_fraction)
    val_patients = set(patients[:n_val])

    val_df = train_val_df[train_val_df["patient_id"].isin(val_patients)].copy()
    train_df = train_val_df[~train_val_df["patient_id"].isin(val_patients)].copy()

    print(f"[splits] train: {len(train_df):>6} images  "
          f"({train_df['patient_id'].nunique():>5} patients)")
    print(f"[splits] val  : {len(val_df):>6} images  "
          f"({val_df['patient_id'].nunique():>5} patients)")
    print(f"[splits] test : {len(test_df):>6} images  "
          f"({test_df['patient_id'].nunique():>5} patients)")

    # Sanity check: no patient leakage
    assert len(set(train_df["patient_id"]) & set(val_df["patient_id"])) == 0
    assert len(set(train_df["patient_id"]) & set(test_df["patient_id"])) == 0
    assert len(set(val_df["patient_id"]) & set(test_df["patient_id"])) == 0
    print("[splits] patient-wise separation verified ✓")

    return train_df, val_df, test_df


# =============================================================================
# 3. DATASET CLASS
# =============================================================================
class NIHChestXray(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14.

    Args:
        df: DataFrame with columns [filename, labels]
        images_dir: directory containing all PNG files (flat structure)
        transform: torchvision transforms applied to PIL image
        apply_clahe: whether to apply CLAHE enhancement before transforms
    """
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform: Optional[transforms.Compose] = None,
        apply_clahe: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.apply_clahe = apply_clahe
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Build a filename -> full path lookup. Supports BOTH layouts:
        # 1. Flat:    <images_dir>/00000001_000.png
        # 2. Kaggle:  <images_dir>/images_001/images/00000001_000.png
        #             <images_dir>/images_002/images/00000002_000.png  ...
        # This lookup is built ONCE at __init__ for O(1) access during training.
        self._path_lookup = {}
        if (self.images_dir / "images_001").exists():
            # Kaggle layout: 12 sub-archives
            for sub in sorted(self.images_dir.glob("images_*")):
                sub_imgs = sub / "images"
                if sub_imgs.exists():
                    for p in sub_imgs.glob("*.png"):
                        self._path_lookup[p.name] = p
        else:
            # Flat layout
            for p in self.images_dir.glob("*.png"):
                self._path_lookup[p.name] = p
        print(f"[dataset] indexed {len(self._path_lookup):,} image files")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._path_lookup.get(row["filename"])
        if img_path is None:
            raise FileNotFoundError(
                f"{row['filename']} not in any indexed images folder"
            )

        # Load as grayscale numpy
        img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)

        # CLAHE contrast enhancement (standard for CXR)
        if self.apply_clahe:
            img = self._clahe.apply(img)

        # Replicate grayscale -> 3 channels for ImageNet-pretrained models
        img = np.stack([img, img, img], axis=-1)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        labels = torch.from_numpy(row["labels"]).float()
        return img, labels


# =============================================================================
# 4. TRANSFORMS
# =============================================================================
def get_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    """
    Training transforms: mild augmentation appropriate for medical images.
    - Horizontal flip: OK for CXR (anatomy is roughly symmetric left-right;
      hearts on left but models still benefit; standard practice)
    - Rotation: small only, real CXRs are upright
    - Color jitter: mild brightness/contrast to simulate acquisition variance
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# 5. DATALOADER FACTORY
# =============================================================================
def make_dataloaders(
    csv_path: str,
    images_dir: str,
    train_val_list: str,
    test_list: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train/val/test DataLoaders + class positive weights for BCE loss.
    Returns: train_loader, val_loader, test_loader, pos_weights
    """
    train_df, val_df, test_df = build_splits(
        csv_path, train_val_list, test_list, val_fraction, seed
    )

    train_ds = NIHChestXray(train_df, images_dir,
                            transform=get_transforms(image_size, train=True))
    val_ds = NIHChestXray(val_df, images_dir,
                          transform=get_transforms(image_size, train=False))
    test_ds = NIHChestXray(test_df, images_dir,
                           transform=get_transforms(image_size, train=False))

    # Class imbalance: compute pos_weight per class for BCEWithLogitsLoss.
    # pos_weight = (# negative samples) / (# positive samples) per class.
    # This upweights rare classes during training.
    label_matrix = np.stack(train_df["labels"].values)  # (N, 14)
    pos_counts = label_matrix.sum(axis=0)
    neg_counts = len(label_matrix) - pos_counts
    pos_weights = torch.from_numpy(neg_counts / (pos_counts + 1e-8)).float()

    print("\n[class balance]")
    for name, pos, weight in zip(DISEASE_LABELS, pos_counts, pos_weights):
        print(f"  {name:20s}  positives={int(pos):>6}  pos_weight={weight:.2f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, pos_weights


# =============================================================================
# 6. QUICK SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    # Change these paths to match your setup
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nih_dataset.py <path_to_nih_dataset_root>")
        print("Expected structure:")
        print("  <root>/Data_Entry_2017.csv")
        print("  <root>/train_val_list.txt")
        print("  <root>/test_list.txt")
        print("  <root>/images/<all PNG files>")
        sys.exit(1)

    root = Path(sys.argv[1])
    train_loader, val_loader, test_loader, pos_weights = make_dataloaders(
        csv_path=str(root / "Data_Entry_2017.csv"),
        images_dir=str(root / "images"),
        train_val_list=str(root / "train_val_list.txt"),
        test_list=str(root / "test_list.txt"),
        batch_size=8,  # small for sanity check
        num_workers=0,
    )

    # Fetch one batch
    imgs, labels = next(iter(train_loader))
    print(f"\n[sanity] batch images shape: {imgs.shape}")
    print(f"[sanity] batch labels shape: {labels.shape}")
    print(f"[sanity] image dtype: {imgs.dtype}, range: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"[sanity] labels dtype: {labels.dtype}, sum per sample: {labels.sum(dim=1).tolist()}")
    print("\n[sanity] ✓ DataLoader works end-to-end")
