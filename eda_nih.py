"""
eda_nih.py
==========
Exploratory Data Analysis for NIH ChestX-ray14.
Produces thesis-ready figures saved to ./figures/:
    1. class_distribution.png       — histogram of disease frequencies
    2. cooccurrence_matrix.png      — which diseases appear together
    3. patient_demographics.png     — age & gender distributions
    4. view_positions.png           — AP vs PA distribution
    5. images_per_patient.png       — follow-up distribution
    6. split_distribution.png       — class balance across train/val/test
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nih_dataset import DISEASE_LABELS, parse_labels, build_splits

sns.set_style("whitegrid")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def fig_class_distribution(df: pd.DataFrame):
    """How many images have each disease label."""
    counts = np.stack(df["labels"].values).sum(axis=0)
    order = np.argsort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar([DISEASE_LABELS[i] for i in order],
                  [counts[i] for i in order], color="steelblue")
    ax.set_ylabel("Number of images")
    ax.set_title("NIH ChestX-ray14: class distribution (multi-label)")
    ax.set_xticklabels([DISEASE_LABELS[i] for i in order],
                       rotation=45, ha="right")
    for bar, val in zip(bars, [counts[i] for i in order]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{int(val)}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print("  ✓ class_distribution.png")


def fig_cooccurrence(df: pd.DataFrame):
    """Co-occurrence matrix — shows multi-label structure."""
    M = np.stack(df["labels"].values)        # (N, 14)
    co = M.T @ M                              # (14, 14)
    # Normalize each row by diagonal: P(col | row)
    diag = np.diag(co).astype(float)
    co_norm = co / (diag[:, None] + 1e-8)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(co_norm, xticklabels=DISEASE_LABELS, yticklabels=DISEASE_LABELS,
                cmap="YlOrRd", annot=True, fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"label": "P(column | row)"}, ax=ax)
    ax.set_title("Disease co-occurrence (row-normalized)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cooccurrence_matrix.png", dpi=150)
    plt.close()
    print("  ✓ cooccurrence_matrix.png")


def fig_demographics(full_df: pd.DataFrame):
    """Age + gender distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ages = pd.to_numeric(full_df["Patient Age"], errors="coerce")
    ages = ages[(ages > 0) & (ages < 120)]  # filter garbage values
    axes[0].hist(ages, bins=40, color="coral", edgecolor="white")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title(f"Patient age distribution  (n={len(ages):,})")

    gender_counts = full_df["Patient Gender"].value_counts()
    axes[1].bar(gender_counts.index, gender_counts.values,
                color=["#5b8def", "#e87a9b"])
    axes[1].set_ylabel("Number of images")
    axes[1].set_title("Gender distribution")
    for i, (k, v) in enumerate(gender_counts.items()):
        axes[1].text(i, v + 500, f"{v:,}", ha="center")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "patient_demographics.png", dpi=150)
    plt.close()
    print("  ✓ patient_demographics.png")


def fig_view_positions(full_df: pd.DataFrame):
    """PA vs AP distribution — important confounder for CXR models."""
    counts = full_df["View Position"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color="seagreen")
    ax.set_ylabel("Number of images")
    ax.set_title("View position distribution")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 500, f"{v:,}", ha="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "view_positions.png", dpi=150)
    plt.close()
    print("  ✓ view_positions.png")


def fig_images_per_patient(full_df: pd.DataFrame):
    """Follow-up distribution — some patients have many X-rays."""
    counts = full_df.groupby("Patient ID").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts, bins=50, color="slateblue", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Images per patient")
    ax.set_ylabel("Number of patients (log scale)")
    ax.set_title(f"Follow-up distribution  "
                 f"(median={int(counts.median())}, "
                 f"max={int(counts.max())})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "images_per_patient.png", dpi=150)
    plt.close()
    print("  ✓ images_per_patient.png")


def fig_split_distribution(train_df, val_df, test_df):
    """Class prevalence across splits — should be roughly balanced."""
    def prevalence(df):
        M = np.stack(df["labels"].values)
        return M.mean(axis=0)

    train_p = prevalence(train_df)
    val_p = prevalence(val_df)
    test_p = prevalence(test_df)

    x = np.arange(len(DISEASE_LABELS))
    width = 0.27

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, train_p, width, label="train", color="steelblue")
    ax.bar(x,         val_p,   width, label="val",   color="orange")
    ax.bar(x + width, test_p,  width, label="test",  color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(DISEASE_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Prevalence (fraction positive)")
    ax.set_title("Class prevalence across splits")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "split_distribution.png", dpi=150)
    plt.close()
    print("  ✓ split_distribution.png")


def main(root: Path):
    csv_path = root / "Data_Entry_2017.csv"
    full_df = pd.read_csv(csv_path)
    full_df["labels"] = full_df["Finding Labels"].apply(parse_labels)

    print(f"[dataset] total images: {len(full_df):,}")
    print(f"[dataset] unique patients: {full_df['Patient ID'].nunique():,}")

    train_df, val_df, test_df = build_splits(
        csv_path=str(csv_path),
        train_val_list=str(root / "train_val_list.txt"),
        test_list=str(root / "test_list.txt"),
    )

    print("\n[figures] generating:")
    fig_class_distribution(train_df)
    fig_cooccurrence(train_df)
    fig_demographics(full_df)
    fig_view_positions(full_df)
    fig_images_per_patient(full_df)
    fig_split_distribution(train_df, val_df, test_df)
    print(f"\nAll figures saved to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python eda_nih.py <path_to_nih_dataset_root>")
        sys.exit(1)
    main(Path(sys.argv[1]))
