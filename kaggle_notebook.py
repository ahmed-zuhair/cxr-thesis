"""
kaggle_notebook.py
==============
Run this ONCE at the top of every Colab notebook in your thesis.

It handles:
- Google Drive mount (persistent storage for dataset + checkpoints)
- GPU verification
- Package installation
- Path configuration

Expected Drive layout after you upload the dataset:
    /content/drive/MyDrive/thesis/
        data/
            Data_Entry_2017.csv
            train_val_list.txt
            test_list.txt
            images/           <- all 112,120 PNG files flat
        checkpoints/
        logs/
        figures/
"""
# This file is meant to be PASTED into a Colab cell, not imported locally.
# Kept as .py for version control in your repo.

COLAB_SETUP = """
# =====================================================================
# CELL 1: Mount Drive + verify GPU + install packages
# =====================================================================
from google.colab import drive
drive.mount('/content/drive')

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"PyTorch version: {torch.__version__}")

# Install packages not pre-installed on Colab
!pip install -q torch_geometric pennylane grad-cam captum pyradiomics

# =====================================================================
# CELL 2: Set paths and verify dataset
# =====================================================================
import os
from pathlib import Path

THESIS_ROOT = Path('/content/drive/MyDrive/thesis')
DATA_ROOT = THESIS_ROOT / 'data'
CHECKPOINT_DIR = THESIS_ROOT / 'checkpoints'
LOG_DIR = THESIS_ROOT / 'logs'
FIG_DIR = THESIS_ROOT / 'figures'

for d in [CHECKPOINT_DIR, LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Verify dataset files exist
required = [
    DATA_ROOT / 'Data_Entry_2017.csv',
    DATA_ROOT / 'train_val_list.txt',
    DATA_ROOT / 'test_list.txt',
    DATA_ROOT / 'images',
]
for p in required:
    assert p.exists(), f"MISSING: {p}"
print(f"Dataset verified at {DATA_ROOT}")

n_images = len(list((DATA_ROOT / 'images').glob('*.png')))
print(f"Found {n_images:,} PNG images")

# =====================================================================
# CELL 3: Clone your thesis code repo (after you set it up on GitHub)
# =====================================================================
# REPLACE with your actual repo URL once you create it
# !git clone https://github.com/YOURUSERNAME/cxr-thesis.git
# %cd cxr-thesis

# =====================================================================
# CELL 4: Load the data module and create DataLoaders
# =====================================================================
import sys
sys.path.insert(0, '/content/cxr-thesis')  # wherever your code lives

from nih_dataset import make_dataloaders

train_loader, val_loader, test_loader, pos_weights = make_dataloaders(
    csv_path=str(DATA_ROOT / 'Data_Entry_2017.csv'),
    images_dir=str(DATA_ROOT / 'images'),
    train_val_list=str(DATA_ROOT / 'train_val_list.txt'),
    test_list=str(DATA_ROOT / 'test_list.txt'),
    batch_size=32,
    image_size=224,
    num_workers=2,
)
"""

print(COLAB_SETUP)
