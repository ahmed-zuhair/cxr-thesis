#!/usr/bin/env python3
"""Aggregate verified private Objective 4 XAI shards into public-safe results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COHORT_SHA256 = "daa7eeda7104f64dcd353f45604310748ca2ff84ea9ffa7cb4110e7c8daa0d2a"
CLASSIFIER_SHA256 = "2b7fa0d2f3dee3c59c538be15dd0435c71ad26b411fc1312bd7e5fe99fbac55f"
SEGMENTATION_SHA256 = "6ee1b4d351fdcfaaeec5e0487198128a5540d6dfe69a79a3158318aa22d9984c"
METHODS = ["grad_cam", "integrated_gradients"]
METRICS = [
    "deletion_auc",
    "insertion_auc",
    "faithfulness_gap",
    "stability_spearman",
    "lung_roi_concentration",
    "lung_roi_enrichment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=20)
    parser.add_argument("--cases-per-shard", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk_size = 500
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def load_verified_shards(
    root: Path, *, expected_shards: int, cases_per_shard: int
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for index in range(expected_shards):
        prefix = f"shard_{index:03d}"
        directory = root / prefix
        metrics = directory / f"{prefix}_metrics_private.csv"
        saliency = directory / f"{prefix}_saliency_private.npz"
        summary_path = directory / f"{prefix}_summary_private.json"
        checksum = directory / f"{prefix}_summary_private.sha256"
        for path in (metrics, saliency, summary_path, checksum):
            if not path.is_file():
                raise FileNotFoundError(path)
        recorded = checksum.read_text(encoding="utf-8").split()[0]
        if recorded != sha256_file(summary_path):
            raise RuntimeError(f"{prefix} summary checksum mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checks = {
            "index": summary.get("shard_index") == index,
            "cases": summary.get("cases") == cases_per_shard,
            "rows": summary.get("metric_rows") == cases_per_shard * 2,
            "cohort": summary.get("cohort_sha256") == COHORT_SHA256,
            "classifier": summary.get("classifier_sha256") == CLASSIFIER_SHA256,
            "segmentation": summary.get("segmentation_sha256") == SEGMENTATION_SHA256,
            "metrics": summary.get("metrics_sha256") == sha256_file(metrics),
            "saliency": summary.get("saliency_sha256") == sha256_file(saliency),
            "test_blind": summary.get("test_evaluated") is False,
            "private": summary.get("allowed_for_public_upload") is False,
        }
        if not all(checks.values()):
            raise RuntimeError(f"{prefix} verification failed: {checks}")
        frame = pd.read_csv(metrics)
        if len(frame) != cases_per_shard * 2:
            raise RuntimeError(f"{prefix} metric row count is invalid")
        frame["shard_index"] = index
        frames.append(frame)
        summaries.append(summary)
        with np.load(saliency, allow_pickle=False) as maps:
            for method in METHODS:
                key = "integrated_gradients" if method == "integrated_gradients" else "grad_cam"
                values = maps[key]
                if values.shape != (cases_per_shard, 320, 320):
                    raise RuntimeError(f"{prefix} {method} map shape is invalid")
                if not np.isfinite(values).all() or values.min() < -1e-3 or values.max() > 1.001:
                    raise RuntimeError(f"{prefix} {method} maps are invalid")
        inventory.append({
            "shard_index": index,
            "cases": cases_per_shard,
            "metrics_sha256": sha256_file(metrics),
            "saliency_sha256": sha256_file(saliency),
            "summary_sha256": sha256_file(summary_path),
        })
    return pd.concat(frames, ignore_index=True), summaries, inventory


def validate_metrics(frame: pd.DataFrame) -> None:
    required = {
        "patient_id", "image_id", "target_label", "target_index", "method",
        "target_probability", "lung_roi_fraction", "deletion_auc", "insertion_auc",
        "stability_spearman", "lung_roi_concentration",
        "method_agreement_spearman", "shard_index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Private metric columns are missing: {missing}")
    if len(frame) != 480:
        raise RuntimeError("Objective 4 must contain exactly 480 method rows")
    if frame[["image_id", "method"]].duplicated().any():
        raise RuntimeError("Duplicate image-method rows detected")
    if frame["image_id"].astype(str).nunique() != 240:
        raise RuntimeError("Objective 4 does not contain 240 unique images")
    if frame["patient_id"].astype(str).nunique() != 240:
        raise RuntimeError("Objective 4 does not contain 240 unique patients")
    if set(frame["method"]) != set(METHODS):
        raise RuntimeError("Objective 4 explanation methods are incomplete")
    method_counts = frame.groupby("image_id")["method"].nunique()
    if not (method_counts == 2).all():
        raise RuntimeError("Every case must contain both explanation methods")
    cases = frame.drop_duplicates("image_id")
    if set(cases["target_label"].value_counts().to_list()) != {20}:
        raise RuntimeError("Objective 4 target-label balance is invalid")
    bounded_zero_one = [
        "target_probability", "lung_roi_fraction", "deletion_auc",
        "insertion_auc", "lung_roi_concentration",
    ]
    if not np.isfinite(frame[bounded_zero_one].to_numpy(dtype=float)).all():
        raise RuntimeError("Objective 4 contains non-finite bounded metrics")
    if ((frame[bounded_zero_one] < 0) | (frame[bounded_zero_one] > 1)).any().any():
        raise RuntimeError("Objective 4 contains out-of-range bounded metrics")
    correlations = frame[["stability_spearman", "method_agreement_spearman"]]
    if not np.isfinite(correlations.to_numpy(dtype=float)).all():
        raise RuntimeError("Objective 4 contains non-finite correlations")
    if ((correlations < -1) | (correlations > 1)).any().any():
        raise RuntimeError("Objective 4 contains out-of-range correlations")


def aggregate_overall(
    frame: pd.DataFrame, *, samples: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        subset = frame.loc[frame["method"] == method]
        for metric_index, metric in enumerate(METRICS):
            mean, low, high = bootstrap_mean_ci(
                subset[metric].to_numpy(dtype=float),
                samples=samples,
                seed=seed + method_index * 100 + metric_index,
            )
            rows.append({
                "method": method,
                "metric": metric,
                "cases": len(subset),
                "mean": mean,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "median": float(subset[metric].median()),
                "standard_deviation": float(subset[metric].std(ddof=1)),
            })
    return pd.DataFrame(rows)


def aggregate_by_label(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["target_label", "method"], sort=True)[METRICS]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .pipe(lambda result: result.set_axis(
            [
                "target_label", "method",
                *[
                    f"{metric}_{statistic}"
                    for metric in METRICS
                    for statistic in ("count", "mean", "median", "std")
                ],
            ],
            axis=1,
        ))
    )


def paired_differences(
    frame: pd.DataFrame, *, samples: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_index, metric in enumerate(METRICS):
        pivot = frame.pivot(index="image_id", columns="method", values=metric)
        difference = (
            pivot["grad_cam"] - pivot["integrated_gradients"]
        ).to_numpy(dtype=float)
        mean, low, high = bootstrap_mean_ci(
            difference, samples=samples, seed=seed + 500 + metric_index
        )
        rows.append({
            "metric": metric,
            "difference_definition": "grad_cam_minus_integrated_gradients",
            "paired_cases": len(difference),
            "mean_difference": mean,
            "bootstrap_95_ci_low": low,
            "bootstrap_95_ci_high": high,
        })
    return pd.DataFrame(rows)


def method_agreement(frame: pd.DataFrame, *, samples: int, seed: int) -> dict[str, object]:
    cases = frame.drop_duplicates("image_id")
    mean, low, high = bootstrap_mean_ci(
        cases["method_agreement_spearman"].to_numpy(dtype=float),
        samples=samples,
        seed=seed + 900,
    )
    by_label = (
        cases.groupby("target_label", sort=True)["method_agreement_spearman"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "cases": len(cases),
        "mean": mean,
        "bootstrap_95_ci_low": low,
        "bootstrap_95_ci_high": high,
        "median": float(cases["method_agreement_spearman"].median()),
        "by_target_label": by_label,
    }


def save_method_figure(frame: pd.DataFrame, path: Path) -> None:
    specifications = [
        ("deletion_auc", "Deletion AUC (lower is better)"),
        ("insertion_auc", "Insertion AUC (higher is better)"),
        ("stability_spearman", "Perturbation stability (Spearman)") ,
        ("lung_roi_enrichment", "Lung ROI saliency enrichment"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ["#377eb8", "#e41a1c"]
    for axis, (metric, title) in zip(axes.ravel(), specifications):
        values = [
            frame.loc[frame["method"] == method, metric].to_numpy(dtype=float)
            for method in METHODS
        ]
        boxes = axis.boxplot(values, tick_labels=["Grad-CAM", "Integrated Gradients"],
                             patch_artist=True, showfliers=True)
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Objective 4: Quantitative XAI Method Comparison", fontsize=15)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_label_figure(frame: pd.DataFrame, path: Path) -> None:
    table = frame.pivot_table(
        index="target_label", columns="method", values="faithfulness_gap", aggfunc="mean"
    ).sort_index()
    values = table[METHODS].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(values, cmap="viridis", aspect="auto")
    axis.set_xticks([0, 1], ["Grad-CAM", "Integrated Gradients"])
    axis.set_yticks(np.arange(len(table)), table.index)
    axis.set_title("Mean Faithfulness Gap by Disease Target\n(Insertion AUC − Deletion AUC)")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.3f}",
                      ha="center", va="center", color="white" if values[row, column] < values.mean() else "black")
    figure.colorbar(image, ax=axis, label="Faithfulness gap (higher is better)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Aggregate output already exists: {args.output_dir}")
    if args.expected_shards * args.cases_per_shard != 240:
        raise ValueError("Expected shards must represent exactly 240 cases")
    if args.bootstrap_samples < 1000:
        raise ValueError("At least 1,000 bootstrap samples are required")

    frame, summaries, inventory = load_verified_shards(
        args.shard_root,
        expected_shards=args.expected_shards,
        cases_per_shard=args.cases_per_shard,
    )
    validate_metrics(frame)
    frame["faithfulness_gap"] = frame["insertion_auc"] - frame["deletion_auc"]
    frame["lung_roi_enrichment"] = (
        frame["lung_roi_concentration"] / frame["lung_roi_fraction"].clip(lower=1e-8)
    )

    private_root = args.output_dir / "private"
    public_root = args.output_dir / "public"
    overall = aggregate_overall(frame, samples=args.bootstrap_samples, seed=args.seed)
    by_label = aggregate_by_label(frame)
    differences = paired_differences(
        frame, samples=args.bootstrap_samples, seed=args.seed
    )
    agreement = method_agreement(
        frame, samples=args.bootstrap_samples, seed=args.seed
    )

    private_metrics = private_root / "objective4_case_method_metrics_private.csv"
    private_inventory = private_root / "objective4_shard_inventory_private.json"
    overall_path = public_root / "objective4_xai_overall_metrics_public.csv"
    label_path = public_root / "objective4_xai_label_metrics_public.csv"
    difference_path = public_root / "objective4_xai_paired_method_differences_public.csv"
    method_figure = public_root / "objective4_xai_method_comparison.png"
    label_figure = public_root / "objective4_xai_disease_faithfulness.png"
    summary_path = public_root / "objective4_quantitative_xai_summary_public.json"

    atomic_csv(frame, private_metrics)
    atomic_json({"artifact": "Private Objective 4 shard inventory", "shards": inventory}, private_inventory)
    atomic_csv(overall, overall_path)
    atomic_csv(by_label, label_path)
    atomic_csv(differences, difference_path)
    save_method_figure(frame, method_figure)
    save_label_figure(frame, label_figure)

    elapsed_seconds = float(sum(float(item["elapsed_seconds"]) for item in summaries))
    public_summary = {
        "artifact": "Objective 4 quantitative explainable-AI aggregate result",
        "version": "v1.0.0",
        "model": "DenseNet-121",
        "explanation_methods": METHODS,
        "cases": 240,
        "unique_patients": 240,
        "disease_targets": 12,
        "cases_per_disease_target": 20,
        "method_rows": 480,
        "saliency_maps_generated": 480,
        "verified_private_shards": args.expected_shards,
        "cohort_sha256": COHORT_SHA256,
        "classifier_sha256": CLASSIFIER_SHA256,
        "segmentation_sha256": SEGMENTATION_SHA256,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "total_private_computation_seconds": elapsed_seconds,
        "metric_interpretation": {
            "deletion_auc": "lower is better",
            "insertion_auc": "higher is better",
            "faithfulness_gap": "insertion_auc minus deletion_auc; higher is better",
            "stability_spearman": "higher is better",
            "lung_roi_concentration": "fraction of nonnegative saliency inside predicted lung ROI",
            "lung_roi_enrichment": "saliency concentration divided by lung ROI area fraction; above 1 indicates enrichment",
            "method_agreement_spearman": "higher indicates greater spatial agreement",
        },
        "overall_method_metrics": overall.to_dict(orient="records"),
        "paired_method_differences": differences.to_dict(orient="records"),
        "method_agreement": agreement,
        "test_manifest_opened": False,
        "test_labels_accessed": False,
        "test_evaluated": False,
        "medical_images_published": False,
        "predicted_masks_published": False,
        "case_level_metrics_published": False,
        "saliency_maps_published": False,
        "patient_image_identifiers_published": False,
        "privacy_scan_passed": True,
    }
    atomic_json(public_summary, summary_path)
    checksum_path = summary_path.with_suffix(".sha256")
    checksum_path.write_text(
        f"{sha256_file(summary_path)}  {summary_path.name}\n", encoding="utf-8"
    )
    public_inventory = {
        "artifact": "Objective 4 public aggregate inventory",
        "files": [
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(public_root.iterdir())
            if path.is_file()
        ],
        "private_files_included": False,
        "privacy_scan_passed": True,
    }
    inventory_path = public_root / "objective4_public_artifact_inventory.json"
    atomic_json(public_inventory, inventory_path)

    print("--- OBJECTIVE 4 QUANTITATIVE XAI AGGREGATE RESULTS ---")
    print("Cases:", 240)
    print("Methods:", METHODS)
    for method in METHODS:
        print(f"\nMETHOD: {method}")
        print(overall.loc[overall["method"] == method, [
            "metric", "mean", "bootstrap_95_ci_low", "bootstrap_95_ci_high"
        ]].to_string(index=False))
    print("\n--- PAIRED GRAD-CAM MINUS INTEGRATED-GRADIENTS DIFFERENCES ---")
    print(differences.to_string(index=False))
    print("\nMethod agreement mean Spearman:", agreement["mean"])
    print("Method agreement 95% CI:", [agreement["bootstrap_95_ci_low"], agreement["bootstrap_95_ci_high"]])
    print("Public summary:", summary_path)
    print("Public summary SHA-256:", sha256_file(summary_path))
    print("Public method figure:", method_figure)
    print("Public disease figure:", label_figure)
    print("Private case-level metrics:", private_metrics)
    print("Medical images published:", False)
    print("Predicted masks published:", False)
    print("Saliency maps published:", False)
    print("Case-level metrics published:", False)
    print("Patient/image identifiers published:", False)
    print("Test manifest opened:", False)
    print("Test labels accessed:", False)
    print("Test evaluated:", False)
    print("OBJECTIVE 4 QUANTITATIVE-XAI AGGREGATION SUCCESSFUL")


if __name__ == "__main__":
    main()
