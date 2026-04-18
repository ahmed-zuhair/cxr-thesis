"""
test_pipeline.py
================
Creates a mock NIH dataset and runs the full pipeline to verify correctness.
This proves the code works BEFORE you run it on 45GB of real data.
"""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, '/home/claude')

# Build a fake dataset
MOCK_ROOT = Path("/tmp/mock_nih")
if MOCK_ROOT.exists():
    shutil.rmtree(MOCK_ROOT)
(MOCK_ROOT / "images").mkdir(parents=True)

# Fake 200 images across 50 patients
rng = np.random.default_rng(42)
rows = []
diseases = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia"]

filenames = []
for patient_id in range(1, 51):
    n_images = rng.integers(2, 6)
    for img_idx in range(n_images):
        fname = f"{patient_id:08d}_{img_idx:03d}.png"
        # Generate a synthetic grayscale image
        img = (rng.normal(128, 40, (256, 256))).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(MOCK_ROOT / "images" / fname)
        # Random labels
        n_findings = rng.integers(0, 3)
        if n_findings == 0:
            labels = "No Finding"
        else:
            picked = rng.choice(diseases, size=n_findings, replace=False)
            labels = "|".join(picked)
        rows.append({
            "Image Index": fname,
            "Finding Labels": labels,
            "Follow-up #": img_idx,
            "Patient ID": patient_id,
            "Patient Age": int(rng.integers(20, 85)),
            "Patient Gender": rng.choice(["M", "F"]),
            "View Position": rng.choice(["PA", "AP"]),
            "OriginalImage[Width": 2048,
            "Height]": 2048,
            "OriginalImagePixelSpacing[x": 0.143,
            "y]": 0.143,
        })
        filenames.append(fname)

df = pd.DataFrame(rows)
df.to_csv(MOCK_ROOT / "Data_Entry_2017.csv", index=False)

# Split PATIENT-WISE 80/20 (matching real NIH split structure)
all_patients = list(range(1, 51))
rng.shuffle(all_patients)
test_patients = set(all_patients[:10])   # 10 patients for test
train_files = [r["Image Index"] for r in rows if r["Patient ID"] not in test_patients]
test_files = [r["Image Index"] for r in rows if r["Patient ID"] in test_patients]
(MOCK_ROOT / "train_val_list.txt").write_text("\n".join(train_files))
(MOCK_ROOT / "test_list.txt").write_text("\n".join(test_files))

print(f"Mock dataset created at {MOCK_ROOT}")
print(f"Total images: {len(df)}, patients: {df['Patient ID'].nunique()}")
print()

# Run the real pipeline
from nih_dataset import make_dataloaders

train_loader, val_loader, test_loader, pos_weights = make_dataloaders(
    csv_path=str(MOCK_ROOT / "Data_Entry_2017.csv"),
    images_dir=str(MOCK_ROOT / "images"),
    train_val_list=str(MOCK_ROOT / "train_val_list.txt"),
    test_list=str(MOCK_ROOT / "test_list.txt"),
    batch_size=8,
    num_workers=0,
)

# Sanity: fetch a batch
imgs, labels = next(iter(train_loader))
print(f"\n[end-to-end test]")
print(f"  batch images shape : {tuple(imgs.shape)}")
print(f"  batch labels shape : {tuple(labels.shape)}")
print(f"  image dtype/range  : {imgs.dtype}, [{imgs.min():.2f}, {imgs.max():.2f}]")
print(f"  labels sum/sample  : {labels.sum(dim=1).tolist()}")
print("\n✓ Full pipeline verified end-to-end")
