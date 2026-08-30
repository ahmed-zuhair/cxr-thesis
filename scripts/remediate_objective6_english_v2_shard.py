#!/usr/bin/env python3
"""Create one locked private Objective 6 English v2 remediation shard."""

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
    enforce_source_concept_polarity,
    language_marker_scores,
    normalise_translation_text,
    normalized_numbers,
    numbers_preserved,
    private_report_sha256,
    restore_numeric_tokens,
    shield_numeric_tokens,
)


SOURCE_HASHES = {
    "train": "66e40de90481c004d5a6f70de23500ca6ca911e02a6dca3747ec0ac2e2c9e872",
    "development": "bf81df9ac5ed7b1eb9f474bda3feb5be72e46afe3707d0b54cbf9d82ce65eaf7",
}
INITIAL_ENGLISH_HASHES = {
    "train": "7832b5f90244ac7a66f9641272b4fec10b13d982121ab3f4ed775d4f4777df5b",
    "development": "5914172cd12f034be2122f2479169e4d7c5705acd1f933236d30f0496d9198e2",
}
REMEDIATION_PROTOCOL_SHA256 = "af4ed664c9fe56883105a2da78c4fcd9b98010bd323d8716fb0309e4aad21e21"
REMEDIATION_LOCK_SHA256 = "1e0791d0144f46362762ba28bf1c317d3f317ae9c2ba96ce53728d6297fbf9b1"
MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--train-initial-english", type=Path, required=True)
    parser.add_argument("--development-initial-english", type=Path, required=True)
    parser.add_argument("--remediation-protocol", type=Path, required=True)
    parser.add_argument("--remediation-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
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


def load_sources(train_path: Path, development_path: Path) -> dict[str, str]:
    train = pd.read_csv(train_path, low_memory=False)
    development = pd.read_csv(development_path, low_memory=False)
    if len(train) != 23570 or len(development) != 5713:
        raise RuntimeError("Objective 6 remediation cohort sizes changed")
    reports: dict[str, str] = {}
    for value in pd.concat(
        [train["report"], development["report"]], ignore_index=True
    ):
        report = normalise_translation_text(value)
        key = private_report_sha256(report)
        previous = reports.setdefault(key, report)
        if previous != report:
            raise RuntimeError("A private report SHA-256 collision was detected")
    if len(reports) != 18984:
        raise RuntimeError("Objective 6 remediation unique-report count changed")
    return reports


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Remediation shard exists: {args.output_dir}")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid remediation shard index/count")
    protected = {
        args.train_source: SOURCE_HASHES["train"],
        args.development_source: SOURCE_HASHES["development"],
        args.train_initial_english: INITIAL_ENGLISH_HASHES["train"],
        args.development_initial_english: INITIAL_ENGLISH_HASHES["development"],
        args.remediation_protocol: REMEDIATION_PROTOCOL_SHA256,
        args.remediation_lock: REMEDIATION_LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 remediation input changed: {path}")

    # Verify that the failed English manifests remain exactly aligned, but do not
    # use their report text to tune or select the corrected output.
    for source_path, english_path, role in (
        (args.train_source, args.train_initial_english, "train"),
        (
            args.development_source,
            args.development_initial_english,
            "development",
        ),
    ):
        source = pd.read_csv(source_path, low_memory=False)
        english = pd.read_csv(english_path, low_memory=False)
        keys = source["report"].map(private_report_sha256)
        if len(source) != len(english) or not keys.equals(
            english["source_report_sha256"].astype(str)
        ):
            raise RuntimeError(f"Failed English {role} lineage changed")

    reports = load_sources(args.train_source, args.development_source)
    keys = sorted(reports)
    positions = np.array_split(np.arange(len(keys)), args.shard_count)[args.shard_index]
    shard_keys = [keys[int(position)] for position in positions]
    sources = [reports[key] for key in shard_keys]

    import sentencepiece
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for frozen NLLB remediation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, src_lang="spa_Latn"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")

    def translate(batch: list[str]) -> list[str]:
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                do_sample=False,
                num_beams=5,
                max_new_tokens=256,
                early_stopping=True,
            )
        return [
            normalise_translation_text(value)
            for value in tokenizer.batch_decode(generated, skip_special_tokens=True)
        ]

    corrected: list[str] = []
    sentinel_failures = 0
    second_passes = 0
    for start in range(0, len(sources), args.batch_size):
        source_batch = sources[start : start + args.batch_size]
        shielded: list[str] = []
        mappings: list[dict[str, str]] = []
        for source in source_batch:
            protected_source, mapping = shield_numeric_tokens(source)
            shielded.append(protected_source)
            mappings.append(mapping)
        first = translate(shielded)
        restored: list[str] = []
        for value, mapping in zip(first, mappings, strict=True):
            restored_value, missing = restore_numeric_tokens(value, mapping)
            sentinel_failures += len(missing)
            restored.append(restored_value)

        second_indices = [
            index
            for index, value in enumerate(restored)
            if language_marker_scores(value)[1] > language_marker_scores(value)[0]
        ]
        if second_indices:
            second_inputs: list[str] = []
            second_mappings: list[dict[str, str]] = []
            for index in second_indices:
                protected_target, mapping = shield_numeric_tokens(restored[index])
                second_inputs.append(protected_target)
                second_mappings.append(mapping)
            second_outputs = translate(second_inputs)
            for index, value, mapping in zip(
                second_indices, second_outputs, second_mappings, strict=True
            ):
                restored_value, missing = restore_numeric_tokens(value, mapping)
                sentinel_failures += len(missing)
                restored[index] = restored_value
            second_passes += len(second_indices)

        corrected.extend(
            enforce_source_concept_polarity(source, target)
            for source, target in zip(source_batch, restored, strict=True)
        )
        print(
            json.dumps(
                {
                    "remediated_unique_reports": min(
                        start + len(source_batch), len(sources)
                    ),
                    "shard_unique_reports": len(sources),
                    "shard_index": args.shard_index,
                }
            ),
            flush=True,
        )

    number_eligible = 0
    number_failures = 0
    concept_eligible = 0
    concept_matches = 0
    Spanish_dominant = 0
    for source, target in zip(sources, corrected, strict=True):
        number_eligible += int(bool(normalized_numbers(source)))
        number_failures += int(not numbers_preserved(source, target))
        matches, eligible = concept_polarity_counts(source, target)
        concept_matches += matches
        concept_eligible += eligible
        English, Spanish = language_marker_scores(target)
        Spanish_dominant += int(Spanish > English)

    args.output_dir.mkdir(parents=True)
    translations_path = args.output_dir / "remediated_translations_private.csv"
    pd.DataFrame(
        {"source_report_sha256": shard_keys, "english_report": corrected}
    ).to_csv(translations_path, index=False, lineterminator="\n")
    translations_hash = sha256(translations_path)
    translations_path.with_suffix(".csv.sha256").write_text(
        f"{translations_hash}  {translations_path.name}\n", encoding="utf-8"
    )
    summary = {
        "artifact": "Objective 6 private English v2 factual-remediation shard",
        "version": "v2.0.1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "unique_reports": len(sources),
        "nonempty_reports": sum(bool(value) for value in corrected),
        "number_eligible_reports": number_eligible,
        "number_preservation_failures": number_failures,
        "concept_polarity_eligible": concept_eligible,
        "concept_polarity_matches": concept_matches,
        "Spanish_marker_dominant_reports": Spanish_dominant,
        "second_translation_passes": second_passes,
        "numeric_sentinel_restoration_failures": sentinel_failures,
        "translations_sha256": translations_hash,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "software_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentencepiece": sentencepiece.__version__,
        },
        "remediation_protocol_sha256": REMEDIATION_PROTOCOL_SHA256,
        "remediation_lock_sha256": REMEDIATION_LOCK_SHA256,
        "candidate_training_started": False,
        "candidate_generation_started": False,
        "original_validation_opened": False,
        "locked_test_manifest_opened": False,
        "locked_test_reports_accessed": False,
        "locked_test_evaluated": False,
        "case_level_outputs_public": False,
        "public_upload_allowed": False,
    }
    write_json(summary, args.output_dir / "remediation_summary_private.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("OBJECTIVE 6 PRIVATE ENGLISH V2 REMEDIATION SHARD SUCCESSFUL")


if __name__ == "__main__":
    main()
