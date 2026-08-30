#!/usr/bin/env python3
"""Run an aggregate-only diagnostic of Objective 6 English v2 translation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from cxr_thesis.objective6.translation import (
    PAD_CHEST_6,
    english_concept_polarity,
    normalise_translation_text,
    normalized_numbers,
    private_report_sha256,
    spanish_concept_polarity,
)


SOURCE_HASHES = {
    "train": "66e40de90481c004d5a6f70de23500ca6ca911e02a6dca3747ec0ac2e2c9e872",
    "development": "bf81df9ac5ed7b1eb9f474bda3feb5be72e46afe3707d0b54cbf9d82ce65eaf7",
}
PROTOCOL_SHA256 = "a09241aff9cb998c68023c6399b6bbb33b66b96fd61475a9f0949a42f6143f62"
LOCK_SHA256 = "28b677450562e04542d4516b2c94cd8b4fa7f9f1161ffe7b992e845391c8d6f4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--train-english", type=Path, required=True)
    parser.add_argument("--development-english", type=Path, required=True)
    parser.add_argument("--expected-train-english-sha256", required=True)
    parser.add_argument("--expected-development-english-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--final-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def validate_pair(source_path: Path, english_path: Path, role: str) -> pd.DataFrame:
    source = pd.read_csv(source_path, low_memory=False)
    english = pd.read_csv(english_path, low_memory=False)
    if len(source) != len(english):
        raise RuntimeError(f"Objective 6 English v2 {role} row alignment changed")
    required = {"report", "source_report_sha256", "report_language"}
    if not required.issubset(english.columns) or "report" not in source:
        raise ValueError(f"Objective 6 English v2 {role} columns are incomplete")
    expected_keys = source["report"].map(private_report_sha256)
    if not expected_keys.equals(english["source_report_sha256"].astype(str)):
        raise RuntimeError(f"Objective 6 English v2 {role} source hashes do not align")
    if set(english["report_language"].astype(str)) != {"English"}:
        raise RuntimeError(f"Objective 6 English v2 {role} language marker changed")
    return pd.DataFrame(
        {
            "source_report_sha256": expected_keys,
            "source": source["report"].map(normalise_translation_text),
            "target": english["report"].map(normalise_translation_text),
        }
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Diagnostic output exists: {args.output_dir}")
    protected = {
        args.train_source: SOURCE_HASHES["train"],
        args.development_source: SOURCE_HASHES["development"],
        args.train_english: args.expected_train_english_sha256,
        args.development_english: args.expected_development_english_sha256,
        args.protocol: PROTOCOL_SHA256,
        args.final_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 English v2 input changed: {path}")

    aligned = pd.concat(
        [
            validate_pair(args.train_source, args.train_english, "train"),
            validate_pair(
                args.development_source, args.development_english, "development"
            ),
        ],
        ignore_index=True,
    )
    unique = aligned.drop_duplicates("source_report_sha256", keep="first")
    if len(unique) != 18984:
        raise RuntimeError("Objective 6 English v2 unique-report coverage changed")

    concept = {
        label: {"source_mentions": 0, "target_mentions": 0, "polarity_matches": 0}
        for label in PAD_CHEST_6
    }
    number_categories = {
        "eligible_reports": 0,
        "exact_multiset_preserved": 0,
        "target_contains_no_numbers": 0,
        "target_contains_fewer_numbers": 0,
        "equal_count_but_values_changed": 0,
        "other_failure": 0,
    }
    english_markers = re.compile(
        r"\b(the|without|with|and|of|is|are|was|were|lung|heart|pleural)\b",
        flags=re.IGNORECASE,
    )
    spanish_markers = re.compile(
        r"\b(el|la|los|las|sin|con|y|de|se|pulmon|pulmonar|corazon)\b",
        flags=re.IGNORECASE,
    )
    language_counts = {
        "english_marker_dominant": 0,
        "spanish_marker_dominant": 0,
        "tied_or_marker_free": 0,
    }

    for row in unique.itertuples(index=False):
        source_values = spanish_concept_polarity(row.source)
        target_values = english_concept_polarity(row.target)
        for label, polarity in source_values.items():
            concept[label]["source_mentions"] += 1
            if label in target_values:
                concept[label]["target_mentions"] += 1
                concept[label]["polarity_matches"] += int(
                    target_values[label] == polarity
                )

        source_numbers = normalized_numbers(row.source)
        target_numbers = normalized_numbers(row.target)
        if source_numbers:
            number_categories["eligible_reports"] += 1
            remaining = list(target_numbers)
            preserved = True
            for value in source_numbers:
                if value not in remaining:
                    preserved = False
                    break
                remaining.remove(value)
            if preserved:
                number_categories["exact_multiset_preserved"] += 1
            elif not target_numbers:
                number_categories["target_contains_no_numbers"] += 1
            elif len(target_numbers) < len(source_numbers):
                number_categories["target_contains_fewer_numbers"] += 1
            elif len(target_numbers) == len(source_numbers):
                number_categories["equal_count_but_values_changed"] += 1
            else:
                number_categories["other_failure"] += 1

        english_score = len(english_markers.findall(row.target))
        spanish_score = len(spanish_markers.findall(row.target))
        if english_score > spanish_score:
            language_counts["english_marker_dominant"] += 1
        elif spanish_score > english_score:
            language_counts["spanish_marker_dominant"] += 1
        else:
            language_counts["tied_or_marker_free"] += 1

    source_mentions = sum(value["source_mentions"] for value in concept.values())
    target_mentions = sum(value["target_mentions"] for value in concept.values())
    polarity_matches = sum(value["polarity_matches"] for value in concept.values())
    eligible_numbers = number_categories["eligible_reports"]
    preserved_numbers = number_categories["exact_multiset_preserved"]
    diagnostic = {
        "artifact": "Objective 6 English v2 aggregate translation diagnostic",
        "version": "v2.0.0-diagnostic.1",
        "diagnostic_only": True,
        "translation_cases": len(aligned),
        "unique_source_reports": len(unique),
        "clause_aware_concept_lexical_coverage": (
            target_mentions / source_mentions if source_mentions else 0.0
        ),
        "clause_aware_concept_polarity_agreement": (
            polarity_matches / source_mentions if source_mentions else 0.0
        ),
        "concept_diagnostics_by_label": concept,
        "number_diagnostics": number_categories,
        "number_or_measurement_loss_rate": (
            1.0 - preserved_numbers / eligible_numbers if eligible_numbers else 0.0
        ),
        "target_language_marker_diagnostics": language_counts,
        "original_locked_translation_gate_rewritten": False,
        "candidate_training_started": False,
        "candidate_generation_started": False,
        "original_validation_opened": False,
        "locked_test_manifest_opened": False,
        "locked_test_reports_accessed": False,
        "locked_test_evaluated": False,
        "raw_reports_printed": False,
        "case_level_outputs_public": False,
    }
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "objective6_english_v2_translation_diagnostic_private.json"
    digest = write_json(diagnostic, output)
    print(json.dumps(diagnostic, indent=2, sort_keys=True))
    print("Diagnostic SHA-256:", digest)
    print("OBJECTIVE 6 ENGLISH V2 AGGREGATE TRANSLATION DIAGNOSTIC SUCCESSFUL")


if __name__ == "__main__":
    main()
