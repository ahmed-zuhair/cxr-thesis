#!/usr/bin/env python3
"""Run the single locked Objective 6 English v2 remediation with private recovery."""

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

from cxr_thesis.objective6.translation import normalise_translation_text, private_report_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    "train": "66e40de90481c004d5a6f70de23500ca6ca911e02a6dca3747ec0ac2e2c9e872",
    "development": "bf81df9ac5ed7b1eb9f474bda3feb5be72e46afe3707d0b54cbf9d82ce65eaf7",
}
INITIAL_ENGLISH_HASHES = {
    "train": "7832b5f90244ac7a66f9641272b4fec10b13d982121ab3f4ed775d4f4777df5b",
    "development": "5914172cd12f034be2122f2479169e4d7c5705acd1f933236d30f0496d9198e2",
}
PROTOCOL_SHA256 = "af4ed664c9fe56883105a2da78c4fcd9b98010bd323d8716fb0309e4aad21e21"
LOCK_SHA256 = "1e0791d0144f46362762ba28bf1c317d3f317ae9c2ba96ce53728d6297fbf9b1"
MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
SHARD_FILES = (
    "remediated_translations_private.csv",
    "remediated_translations_private.csv.sha256",
    "remediation_summary_private.json",
    "remediation_summary_private.json.sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--train-initial-english", type=Path, required=True)
    parser.add_argument("--development-initial-english", type=Path, required=True)
    parser.add_argument("--remediation-protocol", type=Path, required=True)
    parser.add_argument("--remediation-lock", type=Path, required=True)
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def valid_shard(directory: Path, index: int, count: int) -> bool:
    paths = [directory / name for name in SHARD_FILES]
    if not all(path.is_file() for path in paths):
        return False
    translations = directory / SHARD_FILES[0]
    summary_path = directory / SHARD_FILES[2]
    if sha256(translations) != (directory / SHARD_FILES[1]).read_text(encoding="utf-8").split()[0]:
        return False
    if sha256(summary_path) != (directory / SHARD_FILES[3]).read_text(encoding="utf-8").split()[0]:
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return all((
        summary.get("artifact") == "Objective 6 private English v2 factual-remediation shard",
        summary.get("version") == "v2.0.1",
        summary.get("shard_index") == index,
        summary.get("shard_count") == count,
        summary.get("translations_sha256") == sha256(translations),
        summary.get("remediation_protocol_sha256") == PROTOCOL_SHA256,
        summary.get("remediation_lock_sha256") == LOCK_SHA256,
        summary.get("locked_test_manifest_opened") is False,
        summary.get("locked_test_reports_accessed") is False,
        summary.get("locked_test_evaluated") is False,
        summary.get("public_upload_allowed") is False,
    ))


def protected(path: Path, expected: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"Protected Objective 6 remediation input changed: {path}")


def load_source(path: Path, role: str) -> pd.DataFrame:
    expected = 23570 if role == "train" else 5713
    protected(path, SOURCE_HASHES[role])
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != expected or "report" not in frame:
        raise RuntimeError(f"Objective 6 remediation {role} source structure changed")
    return frame


def finalize_manifest(source: pd.DataFrame, translations: dict[str, str], destination: Path) -> str:
    output = source.copy()
    keys = [private_report_sha256(value) for value in output["report"]]
    missing = sorted(set(keys) - set(translations))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} remediated reports")
    output["source_report_sha256"] = keys
    output["report"] = [translations[key] for key in keys]
    output["report_language"] = "English"
    output["translation_model"] = MODEL_ID
    output["translation_revision"] = MODEL_REVISION
    output["remediation_protocol_sha256"] = PROTOCOL_SHA256
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, lineterminator="\n")
    digest = sha256(destination)
    destination.with_suffix(".csv.sha256").write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return digest


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded")
    if args.shard_count != 30:
        raise RuntimeError("Objective 6 remediation shard count is locked at 30")
    protected(args.train_initial_english, INITIAL_ENGLISH_HASHES["train"])
    protected(args.development_initial_english, INITIAL_ENGLISH_HASHES["development"])
    protected(args.remediation_protocol, PROTOCOL_SHA256)
    protected(args.remediation_lock, LOCK_SHA256)
    train = load_source(args.train_source, "train")
    development = load_source(args.development_source, "development")
    for source_path, english_path in ((args.train_source, args.train_initial_english), (args.development_source, args.development_initial_english)):
        source = pd.read_csv(source_path, low_memory=False)
        english = pd.read_csv(english_path, low_memory=False)
        if len(source) != len(english) or not pd.Series(source["report"].map(private_report_sha256)).equals(english["source_report_sha256"].astype(str)):
            raise RuntimeError("Initial English lineage changed")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    if not bool(api.model_info(args.hf_repo, token=token).private):
        raise RuntimeError("Objective 6 remediation recovery repository must be private")
    remote = set(api.list_repo_files(args.hf_repo, repo_type="model", token=token))
    prefix = args.hf_path.strip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions = {"restored": 0, "generated": 0, "reused": 0}
    for index in range(args.shard_count):
        name = f"shard_{index:03d}"
        local = args.output_dir / "private" / "shards" / name
        if valid_shard(local, index, args.shard_count):
            actions["reused"] += 1
            continue
        remote_root = f"{prefix}/private/shards/{name}"
        expected_remote = {f"{remote_root}/{item}" for item in SHARD_FILES}
        available = expected_remote & remote
        if available and available != expected_remote:
            raise RuntimeError(f"Partial private remediation shard exists: {name}")
        if available == expected_remote:
            local.mkdir(parents=True, exist_ok=True)
            for filename in SHARD_FILES:
                downloaded = Path(hf_hub_download(repo_id=args.hf_repo, filename=f"{remote_root}/{filename}", repo_type="model", token=token, force_download=True))
                shutil.copy2(downloaded, local / filename)
            if not valid_shard(local, index, args.shard_count):
                raise RuntimeError(f"Recovered remediation shard invalid: {name}")
            actions["restored"] += 1
            print(json.dumps({"remediation_shard_restored": index}), flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix=f"objective6_remediation_{name}_") as temp:
            stage = Path(temp) / name
            command = [sys.executable, str(ROOT / "scripts/remediate_objective6_english_v2_shard.py"), "--train-source", str(args.train_source), "--development-source", str(args.development_source), "--train-initial-english", str(args.train_initial_english), "--development-initial-english", str(args.development_initial_english), "--remediation-protocol", str(args.remediation_protocol), "--remediation-lock", str(args.remediation_lock), "--output-dir", str(stage), "--shard-index", str(index), "--shard-count", str(args.shard_count), "--batch-size", str(args.batch_size), "--seed", str(args.seed)]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, command)
            if not valid_shard(stage, index, args.shard_count):
                raise RuntimeError(f"Generated remediation shard invalid: {name}")
            api.create_commit(repo_id=args.hf_repo, repo_type="model", token=token, operations=[CommitOperationAdd(path_in_repo=f"{remote_root}/{filename}", path_or_fileobj=str(stage / filename)) for filename in SHARD_FILES], commit_message=f"recovery: Objective 6 English remediation {name}")
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(stage, local)
            actions["generated"] += 1
            print(json.dumps({"private_remediation_shard_uploaded": index}), flush=True)

    translations: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    for index in range(args.shard_count):
        local = args.output_dir / "private" / "shards" / f"shard_{index:03d}"
        summaries.append(json.loads((local / SHARD_FILES[2]).read_text(encoding="utf-8")))
        frame = pd.read_csv(local / SHARD_FILES[0], keep_default_na=False)
        for row in frame.itertuples(index=False):
            key, report = str(row.source_report_sha256), normalise_translation_text(row.english_report)
            if key in translations and translations[key] != report:
                raise RuntimeError("Conflicting remediated translations recovered")
            translations[key] = report
    expected_keys = {private_report_sha256(value) for value in pd.concat([train["report"], development["report"]], ignore_index=True)}
    if set(translations) != expected_keys:
        raise RuntimeError("Private remediation coverage is incomplete")
    private = args.output_dir / "private"
    train_hash = finalize_manifest(train, translations, private / "v2_train_english_remediated_report_cohort_private.csv")
    development_hash = finalize_manifest(development, translations, private / "v2_development_english_remediated_report_cohort_private.csv")
    unique_reports = sum(int(item["unique_reports"]) for item in summaries)
    nonempty = sum(int(item["nonempty_reports"]) for item in summaries)
    number_eligible = sum(int(item["number_eligible_reports"]) for item in summaries)
    number_failures = sum(int(item["number_preservation_failures"]) for item in summaries)
    concept_eligible = sum(int(item["concept_polarity_eligible"]) for item in summaries)
    concept_matches = sum(int(item["concept_polarity_matches"]) for item in summaries)
    spanish_dominant = sum(int(item["Spanish_marker_dominant_reports"]) for item in summaries)
    gates = {
        "minimum_nonempty_fraction": nonempty / unique_reports >= 0.995,
        "maximum_number_or_measurement_loss_rate": (number_failures / number_eligible if number_eligible else 0.0) <= 0.01,
        "minimum_PadChest6_concept_polarity_agreement": (concept_matches / concept_eligible if concept_eligible else 0.0) >= 0.95,
        "no_Spanish_marker_dominant_reports": spanish_dominant == 0,
    }
    inventory = {
        "artifact": "Objective 6 private English v2 factual-remediation inventory",
        "version": "v2.0.1", "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "unique_source_reports": unique_reports, "shards": args.shard_count, "actions": actions,
        "nonempty_fraction": nonempty / unique_reports,
        "number_or_measurement_loss_rate": number_failures / number_eligible if number_eligible else 0.0,
        "PadChest6_concept_polarity_agreement": concept_matches / concept_eligible if concept_eligible else 0.0,
        "Spanish_marker_dominant_reports": spanish_dominant, "quality_gates": gates,
        "remediation_quality_gate_passed": all(gates.values()),
        "private_english_manifest_sha256": {"train": train_hash, "development": development_hash},
        "protocol_sha256": PROTOCOL_SHA256, "remediation_lock_sha256": LOCK_SHA256,
        "raw_reports_public": False, "private_manifests_public": False,
        "locked_test_manifest_opened": False, "locked_test_reports_accessed": False,
        "locked_test_evaluated": False, "public_upload_allowed": False,
    }
    inventory_path = private / "remediation_inventory_private.json"
    write_json(inventory, inventory_path)
    public = args.output_dir / "public"
    public_summary = {key: value for key, value in inventory.items() if key not in {"actions", "private_english_manifest_sha256"}}
    public_summary["artifact"] = "Objective 6 English v2 factual-remediation quality summary"
    public_summary["case_level_outputs_public"] = False
    write_json(public_summary, public / "objective6_english_v2_remediation_summary_public.json")
    api.create_commit(repo_id=args.hf_repo, repo_type="model", token=token, operations=[CommitOperationAdd(path_in_repo=f"{prefix}/private/{path.name}", path_or_fileobj=str(path)) for path in (private / "v2_train_english_remediated_report_cohort_private.csv", private / "v2_train_english_remediated_report_cohort_private.csv.sha256", private / "v2_development_english_remediated_report_cohort_private.csv", private / "v2_development_english_remediated_report_cohort_private.csv.sha256", inventory_path, inventory_path.with_suffix(".json.sha256"))], commit_message="recovery: finalize Objective 6 English remediation")
    print(json.dumps(inventory, indent=2, sort_keys=True))
    print("OBJECTIVE 6 PRIVATE ENGLISH V2 REMEDIATION WITH RECOVERY SUCCESSFUL")


if __name__ == "__main__":
    main()
