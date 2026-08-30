#!/usr/bin/env python3
"""Translate Objective 6 v2 reports with private, shard-level HF recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from cxr_thesis.objective6.translation import (
    normalise_translation_text,
    private_report_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SHA256 = "66e40de90481c004d5a6f70de23500ca6ca911e02a6dca3747ec0ac2e2c9e872"
DEVELOPMENT_SHA256 = "bf81df9ac5ed7b1eb9f474bda3feb5be72e46afe3707d0b54cbf9d82ce65eaf7"
PROTOCOL_SHA256 = "a09241aff9cb998c68023c6399b6bbb33b66b96fd61475a9f0949a42f6143f62"
LOCK_SHA256 = "28b677450562e04542d4516b2c94cd8b4fa7f9f1161ffe7b992e845391c8d6f4"
MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
ROLE_SPECS = {
    "v2_train": (23570, TRAIN_SHA256, "v2_train_report_cohort_private.csv"),
    "v2_development": (
        5713,
        DEVELOPMENT_SHA256,
        "v2_development_report_cohort_private.csv",
    ),
}
SHARD_FILES = (
    "translations_private.csv",
    "translations_private.csv.sha256",
    "translation_summary_private.json",
    "translation_summary_private.json.sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--final-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--shard-count", type=int, default=30)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def valid_shard(directory: Path, index: int, count: int) -> bool:
    paths = [directory / name for name in SHARD_FILES]
    if not all(path.is_file() for path in paths):
        return False
    translations = directory / "translations_private.csv"
    summary_path = directory / "translation_summary_private.json"
    if sha256(translations) != paths[1].read_text(encoding="utf-8").split()[0]:
        return False
    if sha256(summary_path) != paths[3].read_text(encoding="utf-8").split()[0]:
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return all(
        (
            summary.get("artifact")
            == "Objective 6 private frozen-NLLB English translation shard",
            summary.get("shard_index") == index,
            summary.get("shard_count") == count,
            summary.get("model") == MODEL_ID,
            summary.get("model_revision") == MODEL_REVISION,
            summary.get("translations_sha256") == sha256(translations),
            summary.get("locked_test_manifest_opened") is False,
            summary.get("locked_test_reports_accessed") is False,
            summary.get("locked_test_evaluated") is False,
            summary.get("public_upload_allowed") is False,
        )
    )


def load_source(path: Path, role: str) -> pd.DataFrame:
    expected_cases, expected_hash, _ = ROLE_SPECS[role]
    if not path.is_file() or sha256(path) != expected_hash:
        raise RuntimeError(f"Protected Objective 6 English v2 cohort changed: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != expected_cases or set(frame["split"].astype(str)) != {role}:
        raise RuntimeError(f"Objective 6 English v2 {role} cohort structure changed")
    if "report" not in frame:
        raise ValueError(f"Objective 6 English v2 {role} report column is missing")
    return frame


def finalize_manifest(
    source: pd.DataFrame,
    translations: dict[str, str],
    destination: Path,
) -> str:
    output = source.copy()
    keys = [private_report_sha256(value) for value in output["report"]]
    missing = sorted(set(keys) - set(translations))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} private English translations")
    output["source_report_sha256"] = keys
    output["report"] = [translations[key] for key in keys]
    output["report_language"] = "English"
    output["translation_model"] = MODEL_ID
    output["translation_revision"] = MODEL_REVISION
    if any(not normalise_translation_text(value) for value in output["report"]):
        raise RuntimeError("Final Objective 6 English manifest contains an empty report")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, lineterminator="\n")
    digest = sha256(destination)
    destination.with_suffix(".csv.sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    return digest


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError("Shard count must be positive")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    protected = {
        args.protocol: PROTOCOL_SHA256,
        args.final_lock: LOCK_SHA256,
    }
    for path, expected in protected.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Protected Objective 6 English v2 input changed: {path}")
    train = load_source(args.train_manifest, "v2_train")
    development = load_source(args.development_manifest, "v2_development")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    if not bool(api.model_info(args.hf_repo, token=token).private):
        raise RuntimeError("Objective 6 English translation recovery repository must be private")
    remote_files = set(
        api.list_repo_files(args.hf_repo, repo_type="model", token=token)
    )
    prefix = args.hf_path.strip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions = {"restored": 0, "generated": 0, "reused": 0}

    for index in range(args.shard_count):
        shard_name = f"shard_{index:03d}"
        local = args.output_dir / "private" / "shards" / shard_name
        if valid_shard(local, index, args.shard_count):
            actions["reused"] += 1
            continue
        remote_root = f"{prefix}/private/shards/{shard_name}"
        expected_remote = {f"{remote_root}/{name}" for name in SHARD_FILES}
        available = expected_remote & remote_files
        if available and available != expected_remote:
            raise RuntimeError(f"Partial private translation shard exists: {shard_name}")
        if available == expected_remote:
            if local.exists():
                shutil.rmtree(local)
            local.mkdir(parents=True)
            for name in SHARD_FILES:
                source = Path(
                    hf_hub_download(
                        repo_id=args.hf_repo,
                        filename=f"{remote_root}/{name}",
                        repo_type="model",
                        token=token,
                        force_download=True,
                    )
                )
                shutil.copy2(source, local / name)
            if not valid_shard(local, index, args.shard_count):
                raise RuntimeError(f"Restored translation shard is invalid: {shard_name}")
            actions["restored"] += 1
            print(json.dumps({"translation_shard_restored": index}), flush=True)
            continue

        with tempfile.TemporaryDirectory(prefix="objective6_english_v2_") as directory:
            stage = Path(directory) / shard_name
            command = [
                sys.executable,
                str(ROOT / "scripts/translate_objective6_english_v2_shard.py"),
                "--train-manifest",
                str(args.train_manifest),
                "--development-manifest",
                str(args.development_manifest),
                "--protocol",
                str(args.protocol),
                "--final-lock",
                str(args.final_lock),
                "--output-dir",
                str(stage),
                "--shard-index",
                str(index),
                "--shard-count",
                str(args.shard_count),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
            ]
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command)
            if not valid_shard(stage, index, args.shard_count):
                raise RuntimeError(f"Generated translation shard is invalid: {shard_name}")
            api.create_commit(
                repo_id=args.hf_repo,
                repo_type="model",
                token=token,
                operations=[
                    CommitOperationAdd(
                        path_in_repo=f"{remote_root}/{name}",
                        path_or_fileobj=str(stage / name),
                    )
                    for name in SHARD_FILES
                ],
                commit_message=f"recovery: Objective 6 English v2 {shard_name}",
            )
            if local.exists():
                shutil.rmtree(local)
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(stage, local)
            actions["generated"] += 1
            print(json.dumps({"private_translation_shard_uploaded": index}), flush=True)

    summaries: list[dict[str, Any]] = []
    translations: dict[str, str] = {}
    for index in range(args.shard_count):
        local = args.output_dir / "private" / "shards" / f"shard_{index:03d}"
        summary = json.loads(
            (local / "translation_summary_private.json").read_text(encoding="utf-8")
        )
        summaries.append(summary)
        frame = pd.read_csv(local / "translations_private.csv", keep_default_na=False)
        for row in frame.itertuples(index=False):
            key = str(row.source_report_sha256)
            report = normalise_translation_text(row.english_report)
            previous = translations.setdefault(key, report)
            if previous != report:
                raise RuntimeError("Conflicting private translations were recovered")

    expected_keys = {
        private_report_sha256(report)
        for report in pd.concat(
            [train["report"], development["report"]], ignore_index=True
        )
    }
    if set(translations) != expected_keys:
        raise RuntimeError("Private translation coverage is incomplete")

    private = args.output_dir / "private"
    train_path = private / "v2_train_english_report_cohort_private.csv"
    development_path = private / "v2_development_english_report_cohort_private.csv"
    final_hashes = {
        "v2_train_english": finalize_manifest(train, translations, train_path),
        "v2_development_english": finalize_manifest(
            development, translations, development_path
        ),
    }
    unique_reports = sum(int(item["unique_reports"]) for item in summaries)
    nonempty = sum(int(item["nonempty_translations"]) for item in summaries)
    number_failures = sum(
        int(item["number_preservation_failures"]) for item in summaries
    )
    number_eligible = sum(int(item["number_eligible_reports"]) for item in summaries)
    concept_matches = sum(int(item["concept_polarity_matches"]) for item in summaries)
    concept_eligible = sum(int(item["concept_polarity_eligible"]) for item in summaries)
    software_versions = {
        json.dumps(item.get("software_versions", {}), sort_keys=True)
        for item in summaries
    }
    if len(software_versions) != 1:
        raise RuntimeError("Translation shards used inconsistent software versions")
    locked_software_versions = json.loads(software_versions.pop())
    nonempty_fraction = nonempty / unique_reports if unique_reports else 0.0
    number_loss_rate = number_failures / number_eligible if number_eligible else 0.0
    concept_agreement = concept_matches / concept_eligible if concept_eligible else 0.0
    gates = {
        "minimum_nonempty_fraction": nonempty_fraction >= 0.995,
        "maximum_number_or_measurement_loss_rate": number_loss_rate <= 0.01,
        "minimum_PadChest6_concept_polarity_agreement": concept_agreement >= 0.95,
    }
    quality_passed = all(gates.values())
    shard_hashes = {
        f"shard_{index:03d}": summaries[index]["translations_sha256"]
        for index in range(args.shard_count)
    }
    inventory = {
        "artifact": "Objective 6 private English v2 translation inventory",
        "version": "v2.0.0",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "software_versions": locked_software_versions,
        "v2_training_cases": len(train),
        "v2_development_cases": len(development),
        "unique_source_reports": unique_reports,
        "shards": args.shard_count,
        "actions": actions,
        "nonempty_fraction": nonempty_fraction,
        "number_or_measurement_loss_rate": number_loss_rate,
        "PadChest6_concept_polarity_agreement": concept_agreement,
        "quality_gates": gates,
        "translation_quality_gate_passed": quality_passed,
        "private_english_manifest_sha256": final_hashes,
        "translation_shard_sha256": shard_hashes,
        "protocol_sha256": PROTOCOL_SHA256,
        "final_lock_sha256": LOCK_SHA256,
        "raw_Spanish_reports_public": False,
        "raw_English_reports_public": False,
        "private_manifests_public": False,
        "patient_or_image_identifiers_public": False,
        "original_validation_opened": False,
        "locked_test_manifest_opened": False,
        "locked_test_reports_accessed": False,
        "locked_test_evaluated": False,
        "public_upload_allowed": False,
    }
    inventory_path = private / "translation_inventory_private.json"
    write_json(inventory, inventory_path)
    public_summary = {
        key: value
        for key, value in inventory.items()
        if key
        not in {
            "private_english_manifest_sha256",
            "translation_shard_sha256",
            "actions",
        }
    }
    public_summary.update(
        {
            "artifact": "Objective 6 English v2 translation quality summary",
            "case_level_translations_public": False,
            "private_manifest_hashes_public": False,
        }
    )
    public_path = (
        args.output_dir / "public" / "objective6_english_v2_translation_summary_public.json"
    )
    public_hash = write_json(public_summary, public_path)
    inventory["public_summary_sha256"] = public_hash
    write_json(inventory, inventory_path)

    final_files = [
        train_path,
        train_path.with_suffix(".csv.sha256"),
        development_path,
        development_path.with_suffix(".csv.sha256"),
        inventory_path,
        inventory_path.with_suffix(".json.sha256"),
    ]
    api.create_commit(
        repo_id=args.hf_repo,
        repo_type="model",
        token=token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{prefix}/private/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in final_files
        ],
        commit_message="recovery: finalize Objective 6 English v2 translation",
    )

    print(json.dumps(inventory, indent=2, sort_keys=True))
    print("OBJECTIVE 6 PRIVATE ENGLISH V2 TRANSLATION WITH RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()
