#!/usr/bin/env python3
"""Generate label-blind locked-test graph shards with private HF recovery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from generate_objective2_graph_shards import (
    atomic_json,
    build_archive,
    clean_orphan_graphs,
    complete_local_shard,
    restore_archive,
    run_generator,
    sha256_file,
    validate_graphs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "objective1" / "default.yaml",
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-test-cases", type=int, default=5_000)
    parser.add_argument("--expected-test-patients", type=int, default=541)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def read_identity_only_test_manifest(
    path: Path,
    *,
    expected_hash: str,
    expected_cases: int,
    expected_patients: int,
) -> pd.DataFrame:
    """Read test identities and clinical fields without loading label values."""

    if sha256_file(path) != expected_hash:
        raise RuntimeError("Locked-test manifest SHA-256 does not match")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    label_columns = [column for column in columns if column.startswith("label_")]
    if not label_columns:
        raise RuntimeError("Locked-test manifest has no label columns to protect")
    identity_columns = [column for column in columns if column not in label_columns]
    required = {
        "dataset",
        "patient_id",
        "study_id",
        "image_id",
        "image_path",
        "modality",
        "view",
        "split",
        "age",
        "sex",
    }
    missing = sorted(required - set(identity_columns))
    if missing:
        raise RuntimeError(f"Locked-test identity fields are missing: {missing}")
    frame = pd.read_csv(
        path,
        usecols=identity_columns,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    if any(column.startswith("label_") for column in frame.columns):
        raise RuntimeError("A locked-test label value entered the graph-generation frame")
    if len(frame) != expected_cases:
        raise RuntimeError(
            f"Locked-test manifest has {len(frame)} cases, expected {expected_cases}"
        )
    if set(frame["split"].astype(str).str.lower()) != {"test"}:
        raise RuntimeError("Locked-test identity frame contains a non-test split")
    patients = int(frame["patient_id"].astype(str).nunique())
    if patients != expected_patients:
        raise RuntimeError(
            f"Locked-test identity frame has {patients} patients, expected {expected_patients}"
        )
    if frame["image_id"].astype(str).duplicated().any():
        raise RuntimeError("Locked-test identity frame contains duplicate image IDs")
    return frame


def main() -> None:
    args = parse_args()
    if args.workers != 1:
        raise ValueError("--workers must remain 1 for deterministic graph generation")
    if args.shard_size <= 0 or args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("Shard, batch, and save sizes must be positive")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("Segmentation checkpoint SHA-256 does not match")

    identity = read_identity_only_test_manifest(
        args.test_manifest,
        expected_hash=args.expected_test_sha256,
        expected_cases=args.expected_test_cases,
        expected_patients=args.expected_test_patients,
    )
    args.work_root.mkdir(parents=True, exist_ok=True)
    identity_path = args.work_root / "locked_test_identity_private.csv"
    identity.to_csv(identity_path, index=False)
    identity_hash = sha256_file(identity_path)

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for private graph recovery")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    info = api.model_info(args.hf_repo, token=token)
    if not bool(info.private):
        raise RuntimeError("Locked-test graph recovery repository must be private")
    remote_files = set(
        api.list_repo_files(args.hf_repo, repo_type="model", token=token)
    )

    args.graph_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.work_root / "downloads"
    cache_root.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, object]] = []
    total_shards = (len(identity) + args.shard_size - 1) // args.shard_size
    print(
        json.dumps(
            {
                "event": "locked_test_graph_shards_start",
                "test_cases": len(identity),
                "test_patients": int(identity["patient_id"].nunique()),
                "test_label_values_accessed": False,
                "test_evaluated": False,
                "total_shards": total_shards,
                "private_recovery_repo": True,
            }
        )
    )

    for shard_index, start in enumerate(range(0, len(identity), args.shard_size)):
        stop = min(start + args.shard_size, len(identity))
        shard_name = f"shard-{shard_index:05d}"
        shard_root = args.work_root / "shards" / shard_name
        shard_root.mkdir(parents=True, exist_ok=True)
        frame = identity.iloc[start:stop].copy()
        manifest_path = shard_root / "manifest.csv"
        frame.to_csv(manifest_path, index=False)
        manifest_hash = sha256_file(manifest_path)
        remote_archive = f"{args.hf_path.strip('/')}/shards/{shard_name}.zip"
        remote_checksum = f"{remote_archive}.sha256"
        archive_path = shard_root / f"{shard_name}.zip"
        checksum_path = shard_root / f"{shard_name}.zip.sha256"
        has_archive = remote_archive in remote_files
        has_checksum = remote_checksum in remote_files
        if has_archive != has_checksum:
            raise RuntimeError(f"Incomplete private recovery pair for {shard_name}")

        remote_expected_hash: str | None = None
        if has_checksum:
            downloaded_checksum = Path(
                hf_hub_download(
                    args.hf_repo,
                    filename=remote_checksum,
                    repo_type="model",
                    token=token,
                    local_dir=cache_root,
                    force_download=True,
                )
            )
            remote_expected_hash = downloaded_checksum.read_text(
                encoding="utf-8"
            ).split()[0]

        action = "local"
        if not complete_local_shard(frame, shard_root, args.graph_root):
            if has_archive:
                downloaded_archive = Path(
                    hf_hub_download(
                        args.hf_repo,
                        filename=remote_archive,
                        repo_type="model",
                        token=token,
                        local_dir=cache_root,
                    )
                )
                if remote_expected_hash is None:
                    raise RuntimeError(f"Missing private checksum for {shard_name}")
                restore_archive(
                    frame,
                    shard_root,
                    args.graph_root,
                    downloaded_archive,
                    remote_expected_hash,
                )
                action = "restored"
            else:
                clean_orphan_graphs(frame, shard_root, args.graph_root)
                run_generator(args, manifest_path, manifest_hash, shard_root)
                if not complete_local_shard(frame, shard_root, args.graph_root):
                    raise RuntimeError(f"Generated shard failed validation: {shard_name}")
                action = "generated"

        metrics = validate_graphs(frame, args.graph_root)
        archive_hash = build_archive(frame, shard_root, args.graph_root, archive_path)
        checksum_path.write_text(
            f"{archive_hash}  {archive_path.name}\n", encoding="utf-8"
        )
        if has_archive:
            if archive_hash != remote_expected_hash:
                raise RuntimeError(
                    f"Local and private remote hashes differ for {shard_name}"
                )
        else:
            api.create_commit(
                repo_id=args.hf_repo,
                repo_type="model",
                token=token,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=remote_archive,
                        path_or_fileobj=str(archive_path),
                    ),
                    CommitOperationAdd(
                        path_in_repo=remote_checksum,
                        path_or_fileobj=str(checksum_path),
                    ),
                ],
                commit_message=f"recovery: add locked-test graph {shard_name}",
            )
            remote_files.update({remote_archive, remote_checksum})
        shard_records.append(
            {
                "shard": shard_name,
                "start": start,
                "stop": stop,
                "cases": len(frame),
                "manifest_sha256": manifest_hash,
                "archive_sha256": archive_hash,
                "action": action,
                **metrics,
            }
        )
        print(
            json.dumps(
                {
                    "event": "locked_test_graph_shard_complete",
                    "index": shard_index + 1,
                    "total": total_shards,
                    "shard": shard_name,
                    "action": action,
                    **metrics,
                }
            )
        )

    index_payload = {
        "artifact": "Private Objective 2 label-blind locked-test graph recovery index",
        "full_test_manifest_sha256": args.expected_test_sha256,
        "identity_manifest_sha256": identity_hash,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "test_cases": len(identity),
        "test_patients": int(identity["patient_id"].nunique()),
        "complete_graphs": len(identity),
        "node_feature_dimension": 7,
        "shard_size": args.shard_size,
        "shards": shard_records,
        "locked_test_label_values_accessed": False,
        "test_evaluated": False,
        "predicted_masks_saved": False,
        "original_medical_images_copied": False,
        "allowed_for_public_upload": False,
    }
    index_path = args.work_root / "private_locked_test_graph_recovery_index.json"
    atomic_json(index_payload, index_path)
    api.upload_file(
        path_or_fileobj=str(index_path),
        path_in_repo=(
            f"{args.hf_path.strip('/')}/private_locked_test_graph_recovery_index.json"
        ),
        repo_id=args.hf_repo,
        repo_type="model",
        token=token,
        commit_message="recovery: finalize Objective 2 locked-test graph index",
    )
    print(json.dumps(index_payload, indent=2, sort_keys=True))
    print("OBJECTIVE 2 PRIVATE LOCKED-TEST GRAPH SHARDS SUCCESSFUL")


if __name__ == "__main__":
    main()
