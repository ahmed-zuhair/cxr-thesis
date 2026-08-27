#!/usr/bin/env python3
"""Restore exact Objective 2 graph shards and cohorts into a fresh runtime."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
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
        description="Restore private Objective 2 graph shards after a fresh runtime"
    )
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-val-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-train-cases", type=int, default=30_000)
    parser.add_argument("--expected-val-cases", type=int, default=5_000)
    parser.add_argument("--expected-shards", type=int, default=35)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksum(path: Path, digest: str) -> None:
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )


def restore_member(bundle: zipfile.ZipFile, member: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with bundle.open(member) as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install huggingface_hub before restoring shards") from error

    api = HfApi(token=token)
    info = api.model_info(args.hf_repo, token=token)
    if not bool(info.private):
        raise RuntimeError("Graph recovery repository must be private")
    prefix = args.hf_path.strip("/")
    remote_files = set(
        api.list_repo_files(args.hf_repo, repo_type="model", token=token)
    )
    index_remote = f"{prefix}/private_graph_recovery_index.json"
    if index_remote not in remote_files:
        raise FileNotFoundError(index_remote)

    download_root = args.work_root / "downloads"
    download_root.mkdir(parents=True, exist_ok=True)
    index_download = Path(
        hf_hub_download(
            args.hf_repo,
            filename=index_remote,
            repo_type="model",
            token=token,
            local_dir=download_root,
            force_download=True,
        )
    )
    index = json.loads(index_download.read_text(encoding="utf-8"))
    expected_total = args.expected_train_cases + args.expected_val_cases
    checks = {
        "train_hash": index.get("train_manifest_sha256") == args.expected_train_sha256,
        "validation_hash": index.get("validation_manifest_sha256") == args.expected_val_sha256,
        "checkpoint_hash": index.get("checkpoint_sha256") == args.expected_checkpoint_sha256,
        "train_cases": int(index.get("train_cases", -1)) == args.expected_train_cases,
        "validation_cases": int(index.get("validation_cases", -1)) == args.expected_val_cases,
        "complete_graphs": int(index.get("complete_graphs", -1)) == expected_total,
        "shards": len(index.get("shards", [])) == args.expected_shards,
        "test_blind": index.get("test_evaluated") is False,
        "no_masks": index.get("predicted_masks_saved") is False,
        "no_images": index.get("original_medical_images_copied") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Private recovery index validation failed: {checks}")

    args.graph_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    shard_frames: list[pd.DataFrame] = []
    expected_graph_names: set[str] = set()
    node_minimum: int | None = None
    node_maximum: int | None = None
    edge_minimum: int | None = None
    edge_maximum: int | None = None

    ordered_shards = sorted(index["shards"], key=lambda item: str(item["shard"]))
    for position, record in enumerate(ordered_shards, start=1):
        shard_name = str(record["shard"])
        archive_remote = f"{prefix}/shards/{shard_name}.zip"
        checksum_remote = f"{archive_remote}.sha256"
        if archive_remote not in remote_files or checksum_remote not in remote_files:
            raise FileNotFoundError(f"Missing remote recovery pair for {shard_name}")
        archive = Path(
            hf_hub_download(
                args.hf_repo,
                filename=archive_remote,
                repo_type="model",
                token=token,
                local_dir=download_root,
            )
        )
        checksum = Path(
            hf_hub_download(
                args.hf_repo,
                filename=checksum_remote,
                repo_type="model",
                token=token,
                local_dir=download_root,
            )
        )
        recorded_hash = checksum.read_text(encoding="utf-8").split()[0]
        actual_hash = sha256_file(archive)
        if recorded_hash != actual_hash or record.get("archive_sha256") != actual_hash:
            raise RuntimeError(f"Private archive SHA-256 mismatch for {shard_name}")

        shard_root = args.work_root / "shards" / shard_name
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError(f"CRC integrity failure in {shard_name}")
            names = set(bundle.namelist())
            required = {"manifest.csv", "audit.csv", "summary.json"}
            graph_members = {name for name in names if name.startswith("graphs/")}
            if not required.issubset(names) or names != required | graph_members:
                raise RuntimeError(f"Unexpected archive contents in {shard_name}")
            frame = pd.read_csv(
                io.BytesIO(bundle.read("manifest.csv")),
                dtype={"patient_id": str, "study_id": str, "image_id": str},
            )
            audit = pd.read_csv(
                io.BytesIO(bundle.read("audit.csv")),
                dtype={"image_id": str},
            )
            if len(frame) != int(record["cases"]):
                raise RuntimeError(f"Manifest case count mismatch in {shard_name}")
            if len(audit) != len(frame) or set(audit["status"].astype(str)) != {"complete"}:
                raise RuntimeError(f"Audit is incomplete in {shard_name}")
            frame_graph_names = {
                f"{safe_graph_name(value)}.npz" for value in frame["image_id"].astype(str)
            }
            if graph_members != {f"graphs/{name}" for name in frame_graph_names}:
                raise RuntimeError(f"Graph membership mismatch in {shard_name}")
            for graph_name in sorted(frame_graph_names):
                target = args.graph_root / graph_name
                restore_member(bundle, f"graphs/{graph_name}", target)
                graph = GraphSample.load(target)
                if graph.x.shape[1] != 7:
                    raise RuntimeError(f"Node dimension mismatch in {target}")
                nodes = int(graph.x.shape[0])
                edges = int(graph.edge_index.shape[1])
                node_minimum = nodes if node_minimum is None else min(node_minimum, nodes)
                node_maximum = nodes if node_maximum is None else max(node_maximum, nodes)
                edge_minimum = edges if edge_minimum is None else min(edge_minimum, edges)
                edge_maximum = edges if edge_maximum is None else max(edge_maximum, edges)
            restore_member(bundle, "manifest.csv", shard_root / "manifest.csv")
            restore_member(bundle, "audit.csv", shard_root / "audit.csv")
            restore_member(bundle, "summary.json", shard_root / "summary.json")
        if expected_graph_names & frame_graph_names:
            raise RuntimeError(f"Duplicate graph identities in {shard_name}")
        expected_graph_names.update(frame_graph_names)
        shard_frames.append(frame)
        print(
            json.dumps(
                {
                    "event": "private_graph_shard_restored",
                    "index": position,
                    "total": args.expected_shards,
                    "shard": shard_name,
                    "graphs": len(frame),
                }
            )
        )

    combined = pd.concat(shard_frames, ignore_index=True)
    train = combined.iloc[: args.expected_train_cases].copy()
    validation = combined.iloc[args.expected_train_cases :].copy()
    if set(train["split"].astype(str).str.lower()) != {"train"}:
        raise RuntimeError("Recovered training rows contain another split")
    if set(validation["split"].astype(str).str.lower()) != {"val"}:
        raise RuntimeError("Recovered validation rows contain another split")
    if set(train["patient_id"].astype(str)) & set(validation["patient_id"].astype(str)):
        raise RuntimeError("Recovered cohorts contain patient leakage")

    args.cohort_root.mkdir(parents=True, exist_ok=True)
    train_path = args.cohort_root / "train_cohort_private.csv"
    validation_path = args.cohort_root / "val_cohort_private.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    train_hash = sha256_file(train_path)
    validation_hash = sha256_file(validation_path)
    if train_hash != args.expected_train_sha256:
        raise RuntimeError("Recovered training cohort SHA-256 does not match")
    if validation_hash != args.expected_val_sha256:
        raise RuntimeError("Recovered validation cohort SHA-256 does not match")
    write_checksum(train_path, train_hash)
    write_checksum(validation_path, validation_hash)

    actual_graph_names = {path.name for path in args.graph_root.glob("*.npz")}
    if actual_graph_names != expected_graph_names or len(actual_graph_names) != expected_total:
        raise RuntimeError("Final recovered graph root does not match the private index")
    local_index = args.work_root / "private_graph_recovery_index.json"
    shutil.copy2(index_download, local_index)
    print(
        json.dumps(
            {
                "recovered_graphs": len(actual_graph_names),
                "recovered_shards": len(shard_frames),
                "training_cases": len(train),
                "validation_cases": len(validation),
                "training_sha256": train_hash,
                "validation_sha256": validation_hash,
                "node_feature_dimension": 7,
                "node_count_range": [node_minimum, node_maximum],
                "edge_count_range": [edge_minimum, edge_maximum],
                "patient_leakage": 0,
                "locked_test_manifest_opened": False,
                "locked_test_labels_accessed": False,
                "test_evaluated": False,
                "predicted_masks_restored": False,
                "original_medical_images_restored": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 2 PRIVATE GRAPH AND COHORT FRESH-RUNTIME RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()
