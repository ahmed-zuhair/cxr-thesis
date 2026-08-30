#!/usr/bin/env python3
"""Generate one private Objective 6 v1.1 validation-report shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cxr_thesis.objective6.data import ReportGenerationDataset, collate_reports
from cxr_thesis.objective6.models import DenseNetTransformerReportGenerator
from cxr_thesis.objective6.text import ReportVocabulary

VARIANT = "clinical_guided_multimodal_v1_1"
VALIDATION_MANIFEST_SHA256 = (
    "829573501a62a2269269486218889e908db586da98ec2c264402c345bac5f2d6"
)
ENHANCEMENT_PROTOCOL_SHA256 = (
    "279e4fe83da6d82afcbcce595b5596980ca970fae958a16264d3b3e5172eb1a1"
)
ENHANCEMENT_LOCK_SHA256 = (
    "b840440da16023c0169eb3f32c0f4ce7a20ecfa34f8f6b6bfa8ef20511aa53e6"
)
CHECKPOINT_SHA256 = (
    "bc6c6c27208b31a597890b2f12abc35a7e0979b80d6b77cd5b2e341d43baf89b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--enhancement-protocol", type=Path, required=True)
    parser.add_argument("--enhancement-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
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
    if args.expected_checkpoint_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("Objective 6 v1.1 checkpoint selection changed")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid Objective 6 v1.1 validation shard index")
    if (
        args.shard_count != 20
        or args.image_size != 320
        or args.maximum_length != 160
    ):
        raise RuntimeError("Locked Objective 6 v1.1 generation option changed")
    protected = {
        args.validation_manifest: VALIDATION_MANIFEST_SHA256,
        args.enhancement_protocol: ENHANCEMENT_PROTOCOL_SHA256,
        args.enhancement_lock: ENHANCEMENT_LOCK_SHA256,
        args.checkpoint: CHECKPOINT_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 v1.1 input changed: {path}")
    protocol = json.loads(args.enhancement_protocol.read_text(encoding="utf-8"))
    lock = json.loads(args.enhancement_lock.read_text(encoding="utf-8"))
    architecture = protocol.get("enhanced_architecture", {})
    decoding = protocol.get("decoding", {})
    if (
        architecture.get("name") != VARIANT
        or decoding.get("method") != "deterministic beam search"
        or decoding.get("beam_width") != 3
        or decoding.get("length_normalization_alpha") != 0.7
        or decoding.get("no_repeat_ngram_size") != 4
        or lock.get("protocol_sha256") != ENHANCEMENT_PROTOCOL_SHA256
        or lock.get("locked_test_evaluated") is not False
    ):
        raise RuntimeError("Objective 6 v1.1 enhancement lock changed")

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
        raise RuntimeError("Objective 6 v1.1 validation shard is empty")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocabulary = ReportVocabulary.from_dict(checkpoint["vocabulary"])
    config = checkpoint["model_config"]
    if (
        checkpoint.get("variant") != VARIANT
        or checkpoint.get("validation_manifest_sha256")
        != VALIDATION_MANIFEST_SHA256
        or checkpoint.get("test_evaluated") is not False
        or config.get("use_clinical") is not True
        or config.get("use_concept_token") is not True
        or config.get("beam_width") != 3
        or config.get("length_normalization_alpha") != 0.7
        or config.get("no_repeat_ngram_size") != 4
        or int(config.get("maximum_length")) != args.maximum_length
    ):
        raise RuntimeError("Objective 6 v1.1 checkpoint metadata changed")
    model = DenseNetTransformerReportGenerator(
        len(vocabulary.tokens), maximum_length=args.maximum_length,
        pretrained=False, freeze_image_encoder=True,
        use_clinical=True, use_concept_token=True,
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
                generated = model.generate_beam(
                    image, clinical, bos_id=vocabulary.bos_id,
                    eos_id=vocabulary.eos_id,
                    maximum_length=args.maximum_length, beam_width=3,
                    length_normalization_alpha=0.7,
                    no_repeat_ngram_size=4,
                )
            generated_reports.extend(
                vocabulary.decode(row.tolist()) for row in generated.cpu()
            )
            del image, clinical, generated
    if len(generated_reports) != len(selected):
        raise RuntimeError("Objective 6 v1.1 generated-report count mismatch")

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
        "artifact": "Objective 6 v1.1 private validation generation shard",
        "variant": VARIANT,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "start_index": start,
        "stop_index_exclusive": stop,
        "cases": len(private),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "enhancement_protocol_sha256": ENHANCEMENT_PROTOCOL_SHA256,
        "enhancement_lock_sha256": ENHANCEMENT_LOCK_SHA256,
        "predictions_sha256": predictions_hash,
        "generated_nonempty_fraction": nonempty,
        "decoding": {
            "method": "deterministic beam search", "beam_width": 3,
            "length_normalization_alpha": 0.7, "no_repeat_ngram_size": 4,
        },
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
        "variant": VARIANT, "shard": args.shard_index,
        "cases": len(private), "nonempty_fraction": nonempty,
        "predictions_sha256": predictions_hash,
        "test_evaluated": False, "private_only": True,
    }, sort_keys=True))
    print("OBJECTIVE 6 V1.1 PRIVATE VALIDATION GENERATION SHARD SUCCESSFUL")


if __name__ == "__main__":
    main()
