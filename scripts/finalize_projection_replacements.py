"""Finalize audited projection replacements in the private cohort workspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.projection_replacements import (
    finalize_projection_replacements,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply rank-first eligible projection replacements privately"
    )
    parser.add_argument("--cohort-workspace", required=True)
    parser.add_argument("--reserve-workspace", required=True)
    parser.add_argument("--transaction-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = finalize_projection_replacements(
        args.cohort_workspace,
        args.reserve_workspace,
        args.transaction_root,
    )
    print("--- PRIVATE PROJECTION REPLACEMENT FINALIZATION ---")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Candidate/patient/image identifiers displayed: False")
    print("Allowed for public upload: False")
    print("PRIVATE PROJECTION REPLACEMENTS FINALIZED")


if __name__ == "__main__":
    main()
