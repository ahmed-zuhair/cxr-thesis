#!/usr/bin/env python3
"""Run a private 12-case real-data smoke test for Objective 4 XAI."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective2.data import ImageClassificationDataset
from cxr_thesis.objective2.models import build_classifier
from cxr_thesis.objective2.training import seed_everything
from cxr_thesis.objective4 import GradCAM, integrated_gradients


PRIMARY_LABELS = [
    "Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass",
    "Consolidation", "Pneumothorax", "Pleural_Thickening",
    "Cardiomegaly", "Emphysema", "Edema", "Fibrosis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--expected-cohort-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/"))
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_checkpoint(path: Path, expected_hash: str) -> dict[str, object]:
    if sha256_file(path) != expected_hash:
        raise RuntimeError("DenseNet checkpoint SHA-256 does not match")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "model": checkpoint.get("model_name") == "densenet121",
        "labels": checkpoint.get("label_names") == PRIMARY_LABELS,
        "test_blind": checkpoint.get("test_evaluated") is False,
        "model_state": "model_state" in checkpoint,
    }
    configuration = dict(checkpoint.get("model_config") or {})
    signature = dict(checkpoint.get("training_signature") or {})
    image_size = int(configuration.get("image_size", signature.get("image_size", 0)))
    checks["image_size"] = image_size == 320
    if not all(checks.values()):
        raise RuntimeError(f"DenseNet checkpoint validation failed: {checks}")
    return checkpoint


def main() -> None:
    args = parse_args()
    if args.ig_steps < 2:
        raise ValueError("Integrated Gradients requires at least two steps")
    if args.output_dir.exists():
        raise FileExistsError(f"Smoke output already exists: {args.output_dir}")
    if sha256_file(args.cohort) != args.expected_cohort_sha256:
        raise RuntimeError("Objective 4 private cohort SHA-256 does not match")

    frame = pd.read_csv(args.cohort)
    required = {
        "patient_id", "image_id", "image_path", "split",
        "xai_target_label", "xai_target_index", "xai_selection_rank",
        *[f"label_{label}" for label in PRIMARY_LABELS],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Objective 4 cohort columns are missing: {missing}")
    if len(frame) != 240 or frame["patient_id"].astype(str).nunique() != 240:
        raise RuntimeError("Objective 4 private cohort identity is invalid")
    if set(frame["split"].astype(str).str.lower()) != {"val"}:
        raise RuntimeError("Objective 4 smoke accepts validation rows only")
    smoke = frame.loc[frame["xai_selection_rank"] == 1].copy()
    smoke = smoke.sort_values("xai_target_index", kind="stable").reset_index(drop=True)
    if len(smoke) != 12 or set(smoke["xai_target_label"]) != set(PRIMARY_LABELS):
        raise RuntimeError("Smoke cohort must contain one case per target label")

    checkpoint = validate_checkpoint(args.checkpoint, args.expected_checkpoint_sha256)
    dropout = float(
        dict(checkpoint.get("model_config") or {}).get(
            "dropout", dict(checkpoint.get("training_signature") or {}).get("dropout", 0.2)
        )
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(
        "densenet121", len(PRIMARY_LABELS), image_size=320,
        pretrained=False, dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    target_layer = model.encoder.features.norm5

    label_columns = [f"label_{label}" for label in PRIMARY_LABELS]
    dataset = ImageClassificationDataset(
        smoke,
        label_columns,
        data_root=args.data_root,
        image_size=320,
        augment=False,
        seed=args.seed,
        output_channels=3,
        normalisation="imagenet",
    )
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        clinical = sample["clinical"].unsqueeze(0).to(device)
        target_index = int(smoke.iloc[index]["xai_target_index"])
        target_label = str(smoke.iloc[index]["xai_target_label"])
        with GradCAM(model, target_layer) as explainer:
            grad_cam, grad_logits = explainer(image, clinical, target_index)
        integrated, ig_logits = integrated_gradients(
            model, image, clinical, target_index, steps=args.ig_steps
        )
        cam_values = grad_cam.detach().float().cpu().numpy().ravel()
        ig_values = integrated.detach().float().cpu().numpy().ravel()
        agreement = float(spearmanr(cam_values, ig_values).statistic)
        if not np.isfinite(agreement):
            agreement = 0.0
        row = {
            "target_label": target_label,
            "target_probability": float(torch.sigmoid(grad_logits[0, target_index])),
            "grad_cam_min": float(grad_cam.min()),
            "grad_cam_max": float(grad_cam.max()),
            "integrated_gradients_min": float(integrated.min()),
            "integrated_gradients_max": float(integrated.max()),
            "method_agreement_spearman": agreement,
            "logit_maximum_absolute_difference": float(
                (grad_logits - ig_logits).abs().max()
            ),
            "grad_cam_finite": bool(torch.isfinite(grad_cam).all()),
            "integrated_gradients_finite": bool(torch.isfinite(integrated).all()),
        }
        rows.append(row)
        print(
            f"Processed {index + 1}/12: {target_label}; "
            f"agreement={agreement:.4f}"
        )
        del image, clinical, grad_cam, integrated, grad_logits, ig_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    checks = {
        "cases": len(rows) == 12,
        "labels": {row["target_label"] for row in rows} == set(PRIMARY_LABELS),
        "grad_cam_finite": all(row["grad_cam_finite"] for row in rows),
        "integrated_gradients_finite": all(
            row["integrated_gradients_finite"] for row in rows
        ),
        "grad_cam_bounded": all(
            0.0 <= row["grad_cam_min"] <= row["grad_cam_max"] <= 1.0
            for row in rows
        ),
        "integrated_gradients_bounded": all(
            0.0 <= row["integrated_gradients_min"]
            <= row["integrated_gradients_max"] <= 1.0
            for row in rows
        ),
        "consistent_logits": all(
            row["logit_maximum_absolute_difference"] <= 1e-5 for row in rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Real-data XAI smoke checks failed: {checks}")

    payload = {
        "artifact": "Objective 4 private real-data XAI smoke",
        "research_result": False,
        "allowed_for_public_upload": False,
        "model": "densenet121",
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "cohort_sha256": args.expected_cohort_sha256,
        "cases": 12,
        "cases_per_target_label": 1,
        "image_size": 320,
        "integrated_gradients_steps": args.ig_steps,
        "target_layer": "encoder.features.norm5",
        "elapsed_seconds": time.perf_counter() - started,
        "checks": checks,
        "diagnostics": rows,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "medical_images_included": False,
        "saliency_maps_saved": False,
        "test_manifest_opened": False,
        "test_labels_accessed": False,
        "test_evaluated": False,
    }
    output = args.output_dir / "private/xai_real_data_smoke_private.json"
    atomic_json(payload, output)
    output_hash = sha256_file(output)
    output.with_suffix(".sha256").write_text(
        f"{output_hash}  {output.name}\n", encoding="utf-8"
    )
    print("--- OBJECTIVE 4 REAL-DATA XAI SMOKE RESULT ---")
    print("Cases processed:", len(rows))
    print("Labels represented:", len({row["target_label"] for row in rows}))
    print("Grad-CAM finite and bounded:", checks["grad_cam_finite"] and checks["grad_cam_bounded"])
    print("Integrated Gradients finite and bounded:", checks["integrated_gradients_finite"] and checks["integrated_gradients_bounded"])
    print("Consistent model logits:", checks["consistent_logits"])
    print("Output SHA-256:", output_hash)
    print("Medical images saved:", False)
    print("Saliency maps saved:", False)
    print("Test manifest opened:", False)
    print("Test labels accessed:", False)
    print("Test evaluated:", False)
    print("Research result:", False)
    print("Allowed for public upload:", False)
    print("OBJECTIVE 4 REAL-DATA XAI SMOKE SUCCESSFUL")


if __name__ == "__main__":
    main()
