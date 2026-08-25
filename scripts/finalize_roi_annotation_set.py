"""Freeze a reviewed private ROI annotation set and create safe aggregates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.annotation_finalization import (
    finalize_reviewed_annotation_set,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze reviewed private ROI masks")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--qc-audit", required=True)
    parser.add_argument("--focused-review-log")
    parser.add_argument(
        "--allow-single-review-flags",
        action="store_true",
        help=(
            "Freeze masks while preserving unresolved review/QC flags. "
            "Intended for a transparently limited locked test, not training data."
        ),
    )
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize_reviewed_annotation_set(
        args.workspace,
        args.role,
        qc_audit_path=args.qc_audit,
        focused_review_path=args.focused_review_log,
        provenance_path=args.provenance,
        output_dir=args.output_dir,
        expected_cases=args.expected_cases,
        allow_single_review_flags=args.allow_single_review_flags,
    )
    print("--- FINAL REVIEWED ANNOTATION-SET SUMMARY ---")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Private manifest SHA-256: {result['private_manifest_sha256']}")
    print(f"Public summary SHA-256: {result['public_summary_sha256']}")
    print("Candidate identifiers displayed: False")
    print("Medical images or masks copied: False")
    print("Private files allowed for public upload: False")
    print("REVIEWED ROI ANNOTATION SET LOCKED")


if __name__ == "__main__":
    main()
