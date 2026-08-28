#!/usr/bin/env python3
"""Freeze Objective 5 calibration and selected candidates before testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_thesis.objective2.data import ImageClassificationDataset
from cxr_thesis.objective2.metrics import (
    multilabel_metrics,
    select_f1_thresholds,
)
from cxr_thesis.objective2.models import build_classifier

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Pneumothorax",
]
PROTOCOL_SHA256 = "f36064954f16f0831739cf048d223bd39aacf833cc86c3dbbde92ff3c7085dfb"
AUROC_REPRODUCTION_TOLERANCE = 1e-6
EXPECTED = {
    "chexpert": {
        "checkpoint_sha256": "edcd5792c57f04bdbef88043a2a11e422b506bdc2f26cd96f13121f6a8029c12",
        "validation_sha256": "cae4dd0a101257b6d49b58d6401d1b94d1d5914dcc71a7629fbe3d853b0e99dd",
        "zero_shot_auroc": 0.7438386410545034,
        "adapted_auroc": 0.7951094857687545,
    },
    "padchest": {
        "checkpoint_sha256": "109db89a723c6e2f24442cb5866bfcf4084e85083936cda91bce3b8ae4365d9d",
        "validation_sha256": "a7958fead60706378d2e731c40fb5df812aebb4a8fd2c6820fc6a040ce12daae",
        "zero_shot_auroc": 0.8737348139716848,
        "adapted_auroc": 0.9038015157119861,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in EXPECTED:
        parser.add_argument(f"--{dataset}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--{dataset}-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def label_columns(frame: pd.DataFrame) -> list[str]:
    available = {str(column).casefold(): str(column) for column in frame.columns}
    columns = []
    for label in LABELS:
        candidates = (f"label_{label}".casefold(), label.casefold())
        matches = [available[value] for value in candidates if value in available]
        if len(matches) != 1:
            raise ValueError(f"Expected one validation column for {label}: {matches}")
        columns.append(matches[0])
    return columns


def binary_cross_entropy_from_logits(
    logits: np.ndarray, targets: np.ndarray, temperature: float
) -> float:
    scaled = logits / float(temperature)
    losses = (
        np.maximum(scaled, 0.0) - scaled * targets + np.log1p(np.exp(-np.abs(scaled)))
    )
    return float(np.mean(losses))


def fit_temperature(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[float, float, float]:
    before = binary_cross_entropy_from_logits(logits, targets, 1.0)
    result = minimize_scalar(
        lambda value: binary_cross_entropy_from_logits(logits, targets, value),
        bounds=(0.05, 10.0),
        method="bounded",
        options={"xatol": 1e-6, "maxiter": 200},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(result.x)
    after = binary_cross_entropy_from_logits(logits, targets, temperature)
    if after > before + 1e-10:
        raise RuntimeError("Temperature scaling increased validation NLL")
    return temperature, before, after


def macro_brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.mean((probabilities - targets) ** 2, axis=0)))


def macro_ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    values = []
    for label in range(probabilities.shape[1]):
        score = 0.0
        for index, (lower, upper) in enumerate(pairwise(boundaries)):
            selected = (probabilities[:, label] >= lower) & (
                probabilities[:, label] < upper
                if index < bins - 1
                else probabilities[:, label] <= upper
            )
            if selected.any():
                score += float(selected.mean()) * abs(
                    float(probabilities[selected, label].mean())
                    - float(targets[selected, label].mean())
                )
        values.append(score)
    return float(np.mean(values))


@torch.no_grad()
def infer_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for batch in loader:
        output = model(
            batch["image"].to(device, non_blocking=True),
            batch["clinical"].to(device, non_blocking=True),
        )
        logits.append(output.cpu().numpy())
        targets.append(batch["labels"].numpy())
    return np.concatenate(logits), np.concatenate(targets)


def main() -> None:
    args = parse_args()
    summary_path = args.output_dir / "objective5_selection_calibration_public.json"
    lock_path = args.output_dir / "FINAL_OBJECTIVE5_SELECTION_LOCK.json"
    if args.output_dir.exists():
        if not summary_path.is_file() or not lock_path.is_file():
            raise RuntimeError(
                "Partial selection-lock output exists; do not overwrite it"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for artifact_path in (summary_path, lock_path):
            checksum_path = artifact_path.with_suffix(".json.sha256")
            if not checksum_path.is_file():
                raise RuntimeError(f"Missing checksum for {artifact_path.name}")
            recorded_hash = checksum_path.read_text(encoding="utf-8").split()[0]
            if recorded_hash != sha256_file(artifact_path):
                raise RuntimeError(f"Checksum mismatch for {artifact_path.name}")
        if lock.get("summary_sha256") != sha256_file(summary_path):
            raise RuntimeError("Existing selection lock does not match its summary")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("OBJECTIVE 5 SELECTION AND CALIBRATION LOCK RESTORED SUCCESSFULLY")
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for dataset, expected in EXPECTED.items():
        checkpoint_path = getattr(args, f"{dataset}_checkpoint")
        validation_path = getattr(args, f"{dataset}_validation")
        if sha256_file(checkpoint_path) != expected["checkpoint_sha256"]:
            raise RuntimeError(f"{dataset} checkpoint SHA-256 does not match")
        if sha256_file(validation_path) != expected["validation_sha256"]:
            raise RuntimeError(f"{dataset} validation SHA-256 does not match")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("label_names") != LABELS:
            raise RuntimeError(f"{dataset} checkpoint label ordering changed")
        if checkpoint.get("test_evaluated") is not False:
            raise RuntimeError(f"{dataset} checkpoint is not test-blind")
        frame = pd.read_csv(validation_path, dtype={"patient_id": str, "image_id": str})
        if len(frame) != 5_000:
            raise RuntimeError(f"{dataset} validation does not contain 5,000 cases")
        columns = label_columns(frame)
        dataset_object = ImageClassificationDataset(
            frame,
            columns,
            data_root=args.data_root,
            image_size=320,
            augment=False,
            output_channels=3,
            normalisation="imagenet",
            horizontal_flip_probability=0.0,
        )
        loader = DataLoader(
            dataset_object,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
        )
        model = build_classifier(
            "densenet121", len(LABELS), image_size=320, pretrained=False, dropout=0.2
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)
        logits, targets = infer_logits(model, loader, device)
        temperature, nll_before, nll_after = fit_temperature(logits, targets)
        uncalibrated = 1.0 / (1.0 + np.exp(-logits))
        calibrated = 1.0 / (1.0 + np.exp(-(logits / temperature)))
        thresholds = select_f1_thresholds(calibrated, targets)
        uncalibrated_metrics = multilabel_metrics(
            uncalibrated, targets, thresholds=checkpoint["validation_thresholds"]
        )
        calibrated_metrics = multilabel_metrics(
            calibrated, targets, thresholds=thresholds
        )
        observed_auroc = float(calibrated_metrics["macro"]["auroc"])
        auroc_reproduction_difference = abs(observed_auroc - expected["adapted_auroc"])
        if auroc_reproduction_difference > AUROC_REPRODUCTION_TOLERANCE:
            raise RuntimeError(
                f"{dataset} reproduced AUROC {observed_auroc} does not match "
                f"the selected result {expected['adapted_auroc']}"
            )
        results[dataset] = {
            "selected_candidate": "adapted",
            "candidate_checkpoint_sha256": expected["checkpoint_sha256"],
            "validation_manifest_sha256": expected["validation_sha256"],
            "validation_cases": 5_000,
            "zero_shot_macro_auroc": expected["zero_shot_auroc"],
            "reported_adapted_macro_auroc": expected["adapted_auroc"],
            "adapted_macro_auroc": observed_auroc,
            "auroc_reproduction_absolute_difference": (auroc_reproduction_difference),
            "auroc_reproduction_tolerance": AUROC_REPRODUCTION_TOLERANCE,
            "adapted_minus_zero_shot_macro_auroc": observed_auroc
            - expected["zero_shot_auroc"],
            "minimum_required_improvement": 0.005,
            "temperature": temperature,
            "validation_nll_before_calibration": nll_before,
            "validation_nll_after_calibration": nll_after,
            "frozen_thresholds": thresholds.tolist(),
            "uncalibrated_validation_macro_brier": macro_brier(uncalibrated, targets),
            "calibrated_validation_macro_brier": macro_brier(calibrated, targets),
            "uncalibrated_validation_macro_ece": macro_ece(uncalibrated, targets),
            "calibrated_validation_macro_ece": macro_ece(calibrated, targets),
            "uncalibrated_validation_metrics": json_safe(uncalibrated_metrics),
            "calibrated_validation_metrics": json_safe(calibrated_metrics),
        }
    summary = {
        "artifact": "Objective 5 selected-candidate and calibration lock",
        "version": "1.0.0",
        "status": "locked before either external-domain test evaluation",
        "protocol_sha256": PROTOCOL_SHA256,
        "labels": LABELS,
        "datasets": results,
        "selection_rule_applied": True,
        "calibration_method": "one scalar temperature per dataset",
        "thresholds_fitted_on_target_validation": True,
        "temperatures_fitted_on_target_validation": True,
        "test_manifests_opened": False,
        "test_labels_accessed": False,
        "test_evaluated": False,
        "test_used_for_model_selection": False,
        "patient_identifiers_published": False,
        "image_identifiers_published": False,
        "case_level_predictions_published": False,
    }
    atomic_json(summary, summary_path)
    summary_hash = sha256_file(summary_path)
    summary_path.with_suffix(".json.sha256").write_text(
        f"{summary_hash}  {summary_path.name}\n", encoding="utf-8"
    )
    lock = {
        "artifact": "Final Objective 5 pre-test selection lock",
        "version": "1.0.0",
        "summary_sha256": summary_hash,
        "selected_candidates": {
            name: result["candidate_checkpoint_sha256"]
            for name, result in results.items()
        },
        "external_test_evaluation_count": 0,
        "test_evaluated": False,
        "immutable": True,
    }
    atomic_json(lock, lock_path)
    lock_hash = sha256_file(lock_path)
    lock_path.with_suffix(".json.sha256").write_text(
        f"{lock_hash}  {lock_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n--- OBJECTIVE 5 FINAL SELECTION LOCK ---")
    print("Summary SHA-256:", summary_hash)
    print("Final-lock SHA-256:", lock_hash)
    print("Test manifests opened:", False)
    print("Test labels accessed:", False)
    print("Test evaluated:", False)
    print("OBJECTIVE 5 SELECTION AND CALIBRATION LOCK SUCCESSFUL")


if __name__ == "__main__":
    main()
