"""Audit completed private ROI annotations and flag focused review cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.annotation_qc import audit_completed_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit completed private ROI masks")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-foreground-fraction", type=float, default=0.05)
    parser.add_argument("--max-foreground-fraction", type=float, default=0.55)
    parser.add_argument("--max-components", type=int, default=5)
    parser.add_argument("--max-change-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_completed_annotations(
        args.workspace,
        args.role,
        args.output_dir,
        min_foreground_fraction=args.min_foreground_fraction,
        max_foreground_fraction=args.max_foreground_fraction,
        max_components=args.max_components,
        max_change_fraction=args.max_change_fraction,
    )
    print("--- PRIVATE ROI ANNOTATION QC SUMMARY ---")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print("Candidate identifiers displayed: False")
    print("Annotation masks displayed: False")
    print("Allowed for public upload: False")
    print("PRIVATE ROI ANNOTATION QC COMPLETE")


if __name__ == "__main__":
    main()
