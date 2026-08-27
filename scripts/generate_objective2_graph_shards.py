#!/usr/bin/env python3
"""Generate Objective 2 train/validation graphs with private HF shard recovery.

The driver deliberately accepts only the frozen training and validation manifests.
It never opens a locked-test manifest. Each completed shard is compressed and
uploaded to a repository that has been verified as private before generation
continues to the next shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.graphs import GraphSample
from cxr_thesis.objective2.graph_generation import safe_graph_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate private, crash-recoverable Objective 2 graph shards"
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
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
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-train-cases", type=int, default=30_000)
    parser.add_argument("--expected-val-cases", type=int, default=5_000)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for a future parallel reader; must remain 1",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_manifest(
    path: Path,
    expected_split: str,
    expected_hash: str,
    expected_cases: int,
) -> pd.DataFrame:
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"{expected_split} manifest SHA-256 does not match")
    frame = pd.read_csv(
        path,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    if len(frame) != expected_cases:
        raise RuntimeError(
            f"{expected_split} manifest has {len(frame)} cases, expected {expected_cases}"
        )
    observed = set(frame["split"].astype(str).str.lower())
    if observed != {expected_split}:
        raise RuntimeError(
            f"{expected_split} manifest contains splits {sorted(observed)}"
        )
    if frame["image_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{expected_split} manifest contains duplicate image IDs")
    return frame


def graph_paths(frame: pd.DataFrame, graph_root: Path) -> list[Path]:
    return [
        graph_root / f"{safe_graph_name(value)}.npz"
        for value in frame["image_id"].astype(str)
    ]


def validate_graphs(frame: pd.DataFrame, graph_root: Path) -> dict[str, int]:
    nodes: list[int] = []
    edges: list[int] = []
    for path in graph_paths(frame, graph_root):
        if not path.is_file():
            raise FileNotFoundError(path)
        graph = GraphSample.load(path)
        if graph.x.shape[1] != 7:
            raise RuntimeError(f"Unexpected node-feature dimension in {path}")
        nodes.append(int(graph.x.shape[0]))
        edges.append(int(graph.edge_index.shape[1]))
    return {
        "graphs": len(nodes),
        "minimum_nodes": min(nodes),
        "maximum_nodes": max(nodes),
        "minimum_edges": min(edges),
        "maximum_edges": max(edges),
    }


def complete_local_shard(
    frame: pd.DataFrame,
    shard_root: Path,
    graph_root: Path,
) -> bool:
    audit_path = shard_root / "audit.csv"
    summary_path = shard_root / "summary.json"
    if not audit_path.is_file() or not summary_path.is_file():
        return False
    audit = pd.read_csv(audit_path, dtype={"image_id": str})
    if len(audit) != len(frame) or set(audit["status"].astype(str)) != {"complete"}:
        return False
    if set(audit["image_id"].astype(str)) != set(frame["image_id"].astype(str)):
        return False
    if not all(path.is_file() for path in graph_paths(frame, graph_root)):
        return False
    validate_graphs(frame, graph_root)
    return True


def clean_orphan_graphs(
    frame: pd.DataFrame,
    shard_root: Path,
    graph_root: Path,
) -> int:
    """Remove only incomplete files belonging to the current deterministic shard."""
    compatible: set[str] = set()
    audit_path = shard_root / "audit.csv"
    if audit_path.is_file():
        audit = pd.read_csv(audit_path, dtype={"image_id": str})
        compatible = set(
            audit.loc[
                audit["status"].astype(str) == "complete", "image_id"
            ].astype(str)
        )
    removed = 0
    for image_id in frame["image_id"].astype(str):
        path = graph_root / f"{safe_graph_name(image_id)}.npz"
        if path.is_file() and image_id not in compatible:
            path.unlink()
            removed += 1
    return removed


def run_generator(
    args: argparse.Namespace,
    manifest: Path,
    manifest_hash: str,
    shard_root: Path,
) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "generate_objective2_graphs.py"),
        "--manifest",
        str(manifest),
        "--checkpoint",
        str(args.checkpoint),
        "--graph-dir",
        str(args.graph_root),
        "--audit-csv",
        str(shard_root / "audit.csv"),
        "--summary-json",
        str(shard_root / "summary.json"),
        "--data-root",
        str(args.data_root),
        "--config",
        str(args.config),
        "--expected-manifest-sha256",
        manifest_hash,
        "--expected-checkpoint-sha256",
        args.expected_checkpoint_sha256,
        "--batch-size",
        str(args.batch_size),
        "--save-every",
        str(args.save_every),
        "--device",
        args.device,
        "--continue-on-error",
    ]
    if (shard_root / "audit.csv").is_file():
        command.append("--resume")
    subprocess.run(command, check=True)


def build_archive(
    frame: pd.DataFrame,
    shard_root: Path,
    graph_root: Path,
    archive: Path,
) -> str:
    def add_file(bundle: zipfile.ZipFile, source: Path, archive_name: str) -> None:
        information = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
        information.compress_type = zipfile.ZIP_DEFLATED
        information.external_attr = 0o100644 << 16
        bundle.writestr(information, source.read_bytes(), compresslevel=6)

    temporary = archive.with_name(f".{archive.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as bundle:
        for name in ("manifest.csv", "audit.csv", "summary.json"):
            add_file(bundle, shard_root / name, name)
        for path in graph_paths(frame, graph_root):
            add_file(bundle, path, f"graphs/{path.name}")
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Private graph shard archive failed its integrity test")
    temporary.replace(archive)
    return sha256_file(archive)


def restore_archive(
    frame: pd.DataFrame,
    shard_root: Path,
    graph_root: Path,
    archive: Path,
    expected_hash: str,
) -> None:
    if sha256_file(archive) != expected_hash:
        raise RuntimeError(f"Downloaded shard hash mismatch: {archive.name}")
    expected_graph_names = {
        f"graphs/{path.name}" for path in graph_paths(frame, graph_root)
    }
    expected_names = {
        "manifest.csv",
        "audit.csv",
        "summary.json",
        *expected_graph_names,
    }
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        if names != expected_names or bundle.testzip() is not None:
            raise RuntimeError(f"Unexpected or corrupt contents in {archive.name}")
        shard_root.mkdir(parents=True, exist_ok=True)
        graph_root.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.csv", "audit.csv", "summary.json"):
            with bundle.open(name) as source, (shard_root / name).open("wb") as target:
                shutil.copyfileobj(source, target)
        for name in sorted(expected_graph_names):
            target_path = graph_root / Path(name).name
            with bundle.open(name) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
    if not complete_local_shard(frame, shard_root, graph_root):
        raise RuntimeError(f"Restored shard did not validate: {archive.name}")


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0 or args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("Shard size, batch size, and save interval must be positive")
    if args.workers != 1:
        raise ValueError("--workers must remain 1 for deterministic generation")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("Frozen adapted segmentation checkpoint SHA-256 does not match")

    train = read_manifest(
        args.train_manifest,
        "train",
        args.expected_train_sha256,
        args.expected_train_cases,
    )
    validation = read_manifest(
        args.val_manifest,
        "val",
        args.expected_val_sha256,
        args.expected_val_cases,
    )
    if set(train["patient_id"].astype(str)) & set(validation["patient_id"].astype(str)):
        raise RuntimeError("Patient leakage exists between training and validation")
    combined = pd.concat([train, validation], ignore_index=True)

    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub before running graph shards") from error
    api = HfApi(token=token)
    info = api.model_info(args.hf_repo, token=token)
    if not bool(info.private):
        raise RuntimeError("Graph recovery repository must be private")
    remote_files = set(
        api.list_repo_files(args.hf_repo, repo_type="model", token=token)
    )

    args.graph_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.work_root / "downloads"
    cache_root.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, object]] = []
    total_shards = (len(combined) + args.shard_size - 1) // args.shard_size
    print(
        json.dumps(
            {
                "event": "full_graph_shards_start",
                "train_cases": len(train),
                "validation_cases": len(validation),
                "locked_test_manifest_opened": False,
                "total_shards": total_shards,
                "shard_size": args.shard_size,
                "private_recovery_repo": True,
                "predicted_masks_saved": False,
            }
        )
    )

    for shard_index, start in enumerate(range(0, len(combined), args.shard_size)):
        stop = min(start + args.shard_size, len(combined))
        shard_name = f"shard-{shard_index:05d}"
        shard_root = args.work_root / "shards" / shard_name
        shard_root.mkdir(parents=True, exist_ok=True)
        frame = combined.iloc[start:stop].copy()
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
            raise RuntimeError(f"Incomplete remote recovery pair for {shard_name}")

        action = "local"
        remote_expected_hash: str | None = None
        if has_checksum:
            remote_checksum_path = Path(
                hf_hub_download(
                    args.hf_repo,
                    filename=remote_checksum,
                    repo_type="model",
                    token=token,
                    local_dir=cache_root,
                    force_download=True,
                )
            )
            remote_expected_hash = remote_checksum_path.read_text(
                encoding="utf-8"
            ).split()[0]
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
                    raise RuntimeError(f"Missing remote checksum for {shard_name}")
                restore_archive(
                    frame,
                    shard_root,
                    args.graph_root,
                    downloaded_archive,
                    remote_expected_hash,
                )
                action = "restored"
            else:
                removed = clean_orphan_graphs(frame, shard_root, args.graph_root)
                print(
                    json.dumps(
                        {
                            "event": "shard_generate",
                            "shard": shard_name,
                            "cases": len(frame),
                            "orphan_graphs_removed": removed,
                        }
                    )
                )
                run_generator(args, manifest_path, manifest_hash, shard_root)
                if not complete_local_shard(frame, shard_root, args.graph_root):
                    raise RuntimeError(
                        f"Generated shard failed validation: {shard_name}"
                    )
                action = "generated"

        metrics = validate_graphs(frame, args.graph_root)
        archive_hash = build_archive(
            frame,
            shard_root,
            args.graph_root,
            archive_path,
        )
        checksum_path.write_text(
            f"{archive_hash}  {archive_path.name}\n", encoding="utf-8"
        )
        if has_archive:
            if archive_hash != remote_expected_hash:
                raise RuntimeError(
                    f"Local and private remote archive hashes differ for {shard_name}"
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
                commit_message=f"recovery: add Objective 2 graph {shard_name}",
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
                    "event": "shard_complete",
                    "index": shard_index + 1,
                    "total": total_shards,
                    "shard": shard_name,
                    "action": action,
                    **metrics,
                }
            )
        )

    index_payload = {
        "artifact": "Private Objective 2 train-validation graph recovery index",
        "train_manifest_sha256": args.expected_train_sha256,
        "validation_manifest_sha256": args.expected_val_sha256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "train_cases": len(train),
        "validation_cases": len(validation),
        "complete_graphs": len(combined),
        "node_feature_dimension": 7,
        "shard_size": args.shard_size,
        "shards": shard_records,
        "locked_test_manifest_opened": False,
        "locked_test_labels_accessed": False,
        "test_evaluated": False,
        "predicted_masks_saved": False,
        "original_medical_images_copied": False,
        "allowed_for_public_upload": False,
    }
    index_path = args.work_root / "private_graph_recovery_index.json"
    atomic_json(index_payload, index_path)
    api.upload_file(
        path_or_fileobj=str(index_path),
        path_in_repo=f"{args.hf_path.strip('/')}/private_graph_recovery_index.json",
        repo_id=args.hf_repo,
        repo_type="model",
        token=token,
        commit_message="recovery: finalize Objective 2 train-validation graph index",
    )
    print(json.dumps(index_payload, indent=2, sort_keys=True))
    print("OBJECTIVE 2 FULL PRIVATE TRAIN-VALIDATION GRAPH SHARDS SUCCESSFUL")


if __name__ == "__main__":
    main()
