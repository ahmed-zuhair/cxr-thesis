#!/usr/bin/env python3
"""Recover the exact private Objective 2 locked-test cohort from the NIH manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective2.cohort_recovery import recover_exact_cohort_bytes


IDENTITY_COLUMNS = ("patient_id", "split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover an exact, label-blind Objective 2 locked-test cohort"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--expected-test-cases", type=int, default=5_000)
    parser.add_argument("--expected-test-patients", type=int, default=541)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.output.exists():
        raise RuntimeError("Locked-test output already exists; refusing to overwrite it")
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError("Full NIH manifest SHA-256 does not match the protected input")

    # Selection identities are loaded without any disease-label columns.
    identity = pd.read_csv(
        args.manifest,
        usecols=list(IDENTITY_COLUMNS),
        dtype={"patient_id": str, "split": str},
    )
    # Full rows are loaded only after the label-blind selection mechanism is fixed.
    # No test label statistics are calculated or displayed.
    full = pd.read_csv(
        args.manifest,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    payload, recovery = recover_exact_cohort_bytes(
        identity,
        full,
        split="test",
        seed=args.seed,
        target_images=args.expected_test_cases,
        expected_patients=args.expected_test_patients,
        expected_sha256=args.expected_test_sha256,
    )
    if hashlib.sha256(payload).hexdigest() != args.expected_test_sha256:
        raise RuntimeError("Recovered payload failed its final protected hash check")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, args.output)
    checksum = args.output.with_suffix(".sha256")
    checksum.write_text(
        f"{args.expected_test_sha256}  {args.output.name}\n",
        encoding="utf-8",
    )
    private_record = args.output.parent / "locked_test_recovery_private.json"
    private_record.write_text(
        json.dumps(
            {
                "artifact": "Objective 2 private locked-test cohort recovery",
                "source_manifest_sha256": manifest_hash,
                "locked_test_sha256": args.expected_test_sha256,
                "locked_test_cases": args.expected_test_cases,
                "locked_test_patients": args.expected_test_patients,
                "role_seed": args.seed,
                **recovery,
                "test_label_statistics_calculated": False,
                "test_label_statistics_displayed": False,
                "allowed_for_public_upload": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("--- EXACT LOCKED-TEST COHORT RECOVERY ---")
    print("Source manifest SHA-256:", manifest_hash)
    print("Locked-test cases:", args.expected_test_cases)
    print("Locked-test patients:", args.expected_test_patients)
    print("Role seed:", args.seed)
    print("Matching deterministic variants:", len(recovery["matching_variants"]))
    print("Recovered locked-test SHA-256:", sha256_file(args.output))
    print("Hash matches:", sha256_file(args.output) == args.expected_test_sha256)
    print("Labels used during selection:", False)
    print("Predictions used during selection:", False)
    print("Risk scores used during selection:", False)
    print("Test label statistics calculated:", False)
    print("Test label statistics displayed:", False)
    print("Allowed for public upload:", False)
    print("EXACT OBJECTIVE 2 LOCKED-TEST COHORT RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()
