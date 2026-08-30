#!/usr/bin/env python3
"""Generate one private Objective 6 validation-report shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cxr_thesis.objective6.data import ReportGenerationDataset, collate_reports
from cxr_thesis.objective6.models import DenseNetTransformerReportGenerator
from cxr_thesis.objective6.text import ReportVocabulary


VALIDATION_MANIFEST_SHA256 = (
    "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
)
VALIDATION_PROTOCOL_SHA256 = (
    "81424c30f1619707325f0a83ef9a6fba3a859743e3b4ee0c33ac68dba6161438"
)
VALIDATION_LOCK_SHA256 = (
    "e48b11cc0af8be0866b873ae91dd5f4c55738b39927d6dec52d2f29cf5f8275a"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("image_only", "multimodal"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-protocol", type=Path, required=True)
    parser.add_argument("--validation-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--maximum-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksum(path: Path) -> str:
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid validation shard index")
    protected = {
        args.validation_manifest: VALIDATION_MANIFEST_SHA256,
        args.validation_protocol: VALIDATION_PROTOCOL_SHA256,
        args.validation_lock: VALIDATION_LOCK_SHA256,
        args.checkpoint: args.expected_checkpoint_sha256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 input changed: {path}")
    protocol = json.loads(args.validation_protocol.read_text(encoding="utf-8"))
    lock = json.loads(args.validation_lock.read_text(encoding="utf-8"))
    candidate = protocol.get("candidate_training_results", {}).get(args.variant, {})
    if (
        candidate.get("checkpoint_sha256") != args.expected_checkpoint_sha256
        or candidate.get("test_evaluated") is not False
        or lock.get("protocol_sha256") != VALIDATION_PROTOCOL_SHA256
        or lock.get("locked_test_evaluated") is not False
    ):
        raise RuntimeError("Objective 6 validation lock or candidate changed")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    validation = pd.read_csv(args.validation_manifest, low_memory=False)
    if len(validation) != 6280 or set(validation["split"].astype(str)) != {"val"}:
        raise RuntimeError("Objective 6 validation cohort changed")
    start = len(validation) * args.shard_index // args.shard_count
    stop = len(validation) * (args.shard_index + 1) // args.shard_count
    selected = validation.iloc[start:stop].copy().reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("Objective 6 validation shard is empty")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocabulary = ReportVocabulary.from_dict(checkpoint["vocabulary"])
    config = checkpoint["model_config"]
    if (
        checkpoint.get("variant") != args.variant
        or checkpoint.get("validation_manifest_sha256") != VALIDATION_MANIFEST_SHA256
        or checkpoint.get("test_evaluated") is not False
        or bool(config.get("use_clinical")) != (args.variant == "multimodal")
        or int(config.get("maximum_length")) != args.maximum_length
    ):
        raise RuntimeError("Objective 6 checkpoint metadata changed")
    model = DenseNetTransformerReportGenerator(
        len(vocabulary.tokens), maximum_length=args.maximum_length,
        pretrained=False, freeze_image_encoder=True,
        use_clinical=args.variant == "multimodal",
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = ReportGenerationDataset(
        selected, vocabulary, image_size=args.image_size,
        maximum_length=args.maximum_length,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=collate_reports,
    )
    generated_reports: list[str] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            clinical = batch["clinical"].to(device, non_blocking=True)
            with torch.amp.autocast(
                "cuda", enabled=not args.no_amp and device.type == "cuda"
            ):
                generated = model.generate(
                    image, clinical, bos_id=vocabulary.bos_id,
                    eos_id=vocabulary.eos_id,
                    maximum_length=args.maximum_length,
                )
            generated_reports.extend(
                vocabulary.decode(row.tolist()) for row in generated.cpu()
            )
            del image, clinical, generated
    if len(generated_reports) != len(selected):
        raise RuntimeError("Objective 6 generated-report count mismatch")

    private = pd.DataFrame({
        "case_code": selected["case_code"].astype(str),
        "patient_id": selected["patient_id"].astype(str),
        "study_id": selected["study_id"].astype(str),
        "reference_report": selected["report"].fillna("").astype(str),
        "generated_report": generated_reports,
        "reference_labels": selected["labels"].fillna("").astype(str),
    })
    args.output_dir.mkdir(parents=True)
    predictions = args.output_dir / "predictions_private.csv"
    private.to_csv(predictions, index=False, lineterminator="\n")
    predictions_hash = write_checksum(predictions)
    nonempty = float(private["generated_report"].str.strip().ne("").mean())
    summary = {
        "artifact": "Objective 6 private validation generation shard",
        "variant": args.variant,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "start_index": start,
        "stop_index_exclusive": stop,
        "cases": len(private),
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "validation_protocol_sha256": VALIDATION_PROTOCOL_SHA256,
        "validation_lock_sha256": VALIDATION_LOCK_SHA256,
        "predictions_sha256": predictions_hash,
        "generated_nonempty_fraction": nonempty,
        "raw_reports_printed": False,
        "public_upload_allowed": False,
        "test_manifest_opened": False,
        "test_reports_accessed": False,
        "test_evaluated": False,
    }
    summary_path = args.output_dir / "shard_summary_private.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_checksum(summary_path)
    print(json.dumps({
        "variant": args.variant, "shard": args.shard_index,
        "cases": len(private), "nonempty_fraction": nonempty,
        "predictions_sha256": predictions_hash,
        "test_evaluated": False, "private_only": True,
    }, sort_keys=True))
    print("OBJECTIVE 6 PRIVATE VALIDATION GENERATION SHARD SUCCESSFUL")


if __name__ == "__main__":
    main()
