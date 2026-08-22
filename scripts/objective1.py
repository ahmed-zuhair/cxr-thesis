"""Command-line entry point for the Objective 1 research pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.config import load_config
from cxr_thesis.objective1.manifest import build_nih_manifest, validate_manifest, write_manifest
from cxr_thesis.objective1.pipeline import run_cxr_manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Objective 1 data-to-graph pipeline")
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-nih-manifest")
    build.add_argument("--metadata", required=True)
    build.add_argument("--train-val-list", required=True)
    build.add_argument("--test-list", required=True)
    build.add_argument("--images-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--val-fraction", type=float, default=0.10)
    build.add_argument("--seed", type=int, default=42)

    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--data-root", default=".")
    validate.add_argument("--require-files", action="store_true")

    extract = commands.add_parser("extract-cxr")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--config", default=str(REPOSITORY_ROOT / "configs" / "objective1" / "default.yaml"))
    extract.add_argument("--data-root", default=".")
    extract.add_argument("--output-root", required=True)
    extract.add_argument("--limit", type=int)
    extract.add_argument(
        "--allow-full-image-roi",
        action="store_true",
        help="Smoke tests only; invalid for final thesis experiments",
    )
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-nih-manifest":
        frame = build_nih_manifest(
            args.metadata,
            args.train_val_list,
            args.test_list,
            args.images_root,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        target = write_manifest(frame, args.output)
        print(json.dumps({"manifest": str(target), **validate_manifest(frame)}, indent=2))
        return
    if args.command == "validate-manifest":
        frame = pd.read_csv(args.manifest, dtype={"patient_id": str, "study_id": str, "image_id": str})
        print(
            json.dumps(
                validate_manifest(frame, require_files=args.require_files, root=args.data_root),
                indent=2,
            )
        )
        return
    frame = pd.read_csv(args.manifest, dtype={"patient_id": str, "study_id": str, "image_id": str})
    table = run_cxr_manifest(
        frame,
        load_config(args.config),
        args.output_root,
        data_root=args.data_root,
        allow_full_image_roi=args.allow_full_image_roi,
        limit=args.limit,
    )
    print(json.dumps({"processed": int(len(table)), "features": str(Path(args.output_root) / "features.csv")}, indent=2))


if __name__ == "__main__":
    main()

