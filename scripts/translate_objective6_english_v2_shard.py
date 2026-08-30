#!/usr/bin/env python3
"""Translate one private Objective 6 English v2 report shard with frozen NLLB."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cxr_thesis.objective6.translation import (
    concept_polarity_counts,
    normalise_translation_text,
    normalized_numbers,
    numbers_preserved,
    private_report_sha256,
)


TRAIN_SHA256 = "66e40de90481c004d5a6f70de23500ca6ca911e02a6dca3747ec0ac2e2c9e872"
DEVELOPMENT_SHA256 = "bf81df9ac5ed7b1eb9f474bda3feb5be72e46afe3707d0b54cbf9d82ce65eaf7"
PROTOCOL_SHA256 = "a09241aff9cb998c68023c6399b6bbb33b66b96fd61475a9f0949a42f6143f62"
LOCK_SHA256 = "28b677450562e04542d4516b2c94cd8b4fa7f9f1161ffe7b992e845391c8d6f4"
MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
SOURCE_LANGUAGE = "spa_Latn"
TARGET_LANGUAGE = "eng_Latn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--final-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--maximum-source-tokens", type=int, default=512)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6242)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: dict[str, Any], path: Path) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def unique_reports(train: pd.DataFrame, development: pd.DataFrame) -> dict[str, str]:
    output: dict[str, str] = {}
    for report in pd.concat(
        [train["report"], development["report"]], ignore_index=True
    ).tolist():
        normalized = normalise_translation_text(report)
        digest = private_report_sha256(normalized)
        previous = output.setdefault(digest, normalized)
        if previous != normalized:
            raise RuntimeError("A normalized-report SHA-256 collision was detected")
    return output


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Translation shard exists: {args.output_dir}")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    if args.beam_width != 5 or args.maximum_new_tokens != 256:
        raise RuntimeError("The locked Objective 6 v2 decoding configuration changed")

    protected = {
        args.train_manifest: TRAIN_SHA256,
        args.development_manifest: DEVELOPMENT_SHA256,
        args.protocol: PROTOCOL_SHA256,
        args.final_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 English v2 input changed: {path}")

    train = pd.read_csv(args.train_manifest, low_memory=False)
    development = pd.read_csv(args.development_manifest, low_memory=False)
    if len(train) != 23570 or len(development) != 5713:
        raise RuntimeError("Objective 6 English v2 cohort sizes changed")
    if "report" not in train or "report" not in development:
        raise ValueError("Private Objective 6 v2 manifests require the report column")

    reports = unique_reports(train, development)
    keys = sorted(reports)
    positions = np.array_split(np.arange(len(keys)), args.shard_count)[args.shard_index]
    shard_keys = [keys[int(position)] for position in positions]
    shard_reports = [reports[key] for key in shard_keys]

    import torch
    import sentencepiece
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for frozen NLLB translation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        src_lang=SOURCE_LANGUAGE,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)
    if forced_bos_token_id is None or forced_bos_token_id < 0:
        raise RuntimeError("Frozen NLLB tokenizer does not contain eng_Latn")

    translations: list[str] = []
    for start in range(0, len(shard_reports), args.batch_size):
        batch = shard_reports[start : start + args.batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.maximum_source_tokens,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                do_sample=False,
                num_beams=args.beam_width,
                max_new_tokens=args.maximum_new_tokens,
                early_stopping=True,
            )
        translations.extend(
            normalise_translation_text(text)
            for text in tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
        completed = min(start + len(batch), len(shard_reports))
        print(
            json.dumps(
                {
                    "translated_unique_reports": completed,
                    "shard_unique_reports": len(shard_reports),
                    "shard_index": args.shard_index,
                }
            ),
            flush=True,
        )

    if len(translations) != len(shard_reports) or any(not text for text in translations):
        raise RuntimeError("Frozen NLLB returned missing or empty translations")

    number_failures = 0
    number_eligible = 0
    concept_matches = 0
    concept_eligible = 0
    for source, translated in zip(shard_reports, translations, strict=True):
        number_eligible += int(bool(normalized_numbers(source)))
        number_failures += int(not numbers_preserved(source, translated))
        matches, eligible = concept_polarity_counts(source, translated)
        concept_matches += matches
        concept_eligible += eligible

    args.output_dir.mkdir(parents=True)
    translations_path = args.output_dir / "translations_private.csv"
    pd.DataFrame(
        {
            "source_report_sha256": shard_keys,
            "english_report": translations,
        }
    ).to_csv(translations_path, index=False, lineterminator="\n")
    translations_hash = sha256(translations_path)
    translations_path.with_suffix(".csv.sha256").write_text(
        f"{translations_hash}  {translations_path.name}\n", encoding="utf-8"
    )
    summary = {
        "artifact": "Objective 6 private frozen-NLLB English translation shard",
        "version": "v2.0.0",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "unique_reports": len(shard_reports),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "software_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentencepiece": sentencepiece.__version__,
        },
        "source_language": SOURCE_LANGUAGE,
        "target_language": TARGET_LANGUAGE,
        "beam_width": args.beam_width,
        "maximum_new_tokens": args.maximum_new_tokens,
        "nonempty_translations": len(translations),
        "number_preservation_failures": number_failures,
        "number_eligible_reports": number_eligible,
        "concept_polarity_matches": concept_matches,
        "concept_polarity_eligible": concept_eligible,
        "translations_sha256": translations_hash,
        "train_manifest_sha256": TRAIN_SHA256,
        "development_manifest_sha256": DEVELOPMENT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "final_lock_sha256": LOCK_SHA256,
        "Spanish_reports_public": False,
        "English_reports_public": False,
        "patient_or_image_identifiers_public": False,
        "original_validation_opened": False,
        "locked_test_manifest_opened": False,
        "locked_test_reports_accessed": False,
        "locked_test_evaluated": False,
        "public_upload_allowed": False,
    }
    write_json(summary, args.output_dir / "translation_summary_private.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("OBJECTIVE 6 PRIVATE ENGLISH TRANSLATION SHARD SUCCESSFUL")


if __name__ == "__main__":
    main()
