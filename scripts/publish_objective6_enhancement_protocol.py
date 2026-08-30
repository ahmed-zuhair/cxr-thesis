#!/usr/bin/env python3
"""Publish the immutable Objective 6 v1.1 enhancement protocol."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.publish_objective6_validation_protocol import (
    ROOT,
    atomic_json,
    deterministic_zip,
    github_request,
    run_git,
    sha256,
)

V1_SUMMARY_SHA256 = "0bec4540a38993e23327cde334a6a73f97c0a10d4d7d736b6e9f82afa86bcc7a"
V1_LOCK_SHA256 = "a35f55328480be74f01ed8f0879796e82792b1b8946464c4a0976892727d031f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-directory", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-title", required=True)
    return parser.parse_args()


def validate(protocol: dict[str, Any], lock: dict[str, Any]) -> None:
    if protocol.get("artifact") != (
        "Objective 6 clinical-guided report-generation enhancement protocol"
    ):
        raise RuntimeError("Unexpected Objective 6 enhancement protocol")
    if protocol.get("version") != "v1.1.0" or protocol.get("status") != (
        "locked after publication of v1 validation results and before v1.1 "
        "implementation, training, generation, or locked-test access"
    ):
        raise RuntimeError("Objective 6 enhancement protocol status changed")
    if (
        protocol.get("motivation", {}).get("v1_validation_summary_sha256")
        != V1_SUMMARY_SHA256
        or protocol.get("motivation", {}).get("v1_validation_lock_sha256")
        != V1_LOCK_SHA256
    ):
        raise RuntimeError("Objective 6 v1 provenance changed")
    advancement = protocol.get("advancement_rule", {})
    if (
        advancement.get("all_conditions_required") is not True
        or advancement.get("additional_enhancement_rounds_allowed") is not False
        or advancement.get("minimum_macro_concept_f1")
        != 0.22707029780859555
        or advancement.get("minimum_CIDEr_D") != 0.7864452646530204
        or advancement.get("maximum_explicit_negation_contradiction_rate") != 0.85
        or advancement.get("minimum_unique_generated_report_fraction") != 0.20
    ):
        raise RuntimeError("Objective 6 enhancement advancement rule changed")
    safety = protocol.get("safety_state", {})
    if any(
        safety.get(field) is not False
        for field in (
            "v1_1_implementation_started", "v1_1_training_started",
            "v1_1_validation_generation_started", "locked_test_manifest_opened",
            "locked_test_reports_accessed", "locked_test_evaluated",
        )
    ):
        raise RuntimeError("Objective 6 enhancement protocol is not pre-training")
    if lock.get("artifact") != (
        "Final Objective 6 v1.1 pre-implementation enhancement lock"
    ):
        raise RuntimeError("Unexpected Objective 6 enhancement final lock")
    if (
        lock.get("immutable") is not True
        or lock.get("enhancement_rounds_allowed") != 1
        or lock.get("enhancement_rounds_completed") != 0
        or lock.get("v1_1_implementation_started") is not False
        or lock.get("v1_1_training_started") is not False
        or lock.get("v1_1_validation_evaluated") is not False
        or lock.get("locked_test_evaluated") is not False
        or lock.get("locked_test_evaluation_count") != 0
    ):
        raise RuntimeError("Objective 6 enhancement final lock changed")


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")
    protocol_source = args.lock_directory / "objective6_enhancement_protocol_public.json"
    protocol_checksum = protocol_source.with_suffix(".json.sha256")
    lock_source = args.lock_directory / "FINAL_OBJECTIVE6_ENHANCEMENT_PROTOCOL_LOCK.json"
    lock_checksum = lock_source.with_suffix(".json.sha256")
    sources = (protocol_source, protocol_checksum, lock_source, lock_checksum)
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol_hash = sha256(protocol_source)
    lock_hash = sha256(lock_source)
    if (
        protocol_hash != args.expected_protocol_sha256
        or lock_hash != args.expected_lock_sha256
    ):
        raise RuntimeError("Objective 6 enhancement protocol hashes changed")
    if protocol_checksum.read_text(encoding="utf-8").split()[0] != protocol_hash:
        raise RuntimeError("Objective 6 enhancement protocol checksum mismatch")
    if lock_checksum.read_text(encoding="utf-8").split()[0] != lock_hash:
        raise RuntimeError("Objective 6 enhancement final-lock checksum mismatch")
    protocol = json.loads(protocol_source.read_text(encoding="utf-8"))
    lock = json.loads(lock_source.read_text(encoding="utf-8"))
    validate(protocol, lock)
    if lock.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Enhancement final lock does not identify protocol")

    result_root = ROOT / args.result_path
    expected_names = {
        *(path.name for path in sources),
        "README.md", "artifact_inventory_public.json",
    }
    if result_root.exists():
        existing = {path.name for path in result_root.iterdir() if path.is_file()}
        if existing != expected_names:
            raise RuntimeError("Partial Objective 6 enhancement publication exists")
    else:
        result_root.mkdir(parents=True)
        for source in sources:
            shutil.copy2(source, result_root / source.name)
        (result_root / "README.md").write_text(
            "# Objective 6 clinical-guided enhancement protocol v1.1\n\n"
            "This protocol preserves the published v1 result and prospectively "
            "locks one clinical-guided enhancement round before implementation, "
            "training, validation generation, or locked-test access. Advancement "
            "requires all predefined clinical, lexical, contradiction, diversity, "
            "and repetition conditions. No further enhancement round is permitted.\n\n"
            "No reports, identifiers, images, manifests, checkpoints, case-level "
            "outputs, or locked-test results are included.\n",
            encoding="utf-8",
        )
        inventory: dict[str, Any] = {
            "artifact": "Objective 6 enhancement-protocol public inventory",
            "protocol_sha256": protocol_hash,
            "final_lock_sha256": lock_hash,
            "private_manifests_included": False,
            "patient_or_image_identifiers_included": False,
            "medical_images_included": False,
            "raw_or_generated_reports_included": False,
            "case_level_outputs_included": False,
            "private_checkpoints_included": False,
            "locked_test_results_included": False,
            "files": {},
        }
        for path in sorted(result_root.iterdir()):
            inventory["files"][path.name] = {
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
        atomic_json(inventory, result_root / "artifact_inventory_public.json")
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.suffix.lower() in {".json", ".md"}
    )
    forbidden = (
        '"patient_id"', '"image_id"', '"image_path"', '"case_code"',
        '"reference_report"', '"generated_report"',
    )
    violations = [value for value in forbidden if value in serialized]
    if violations:
        raise RuntimeError(f"Objective 6 public privacy scan failed: {violations}")
    archive = Path(
        "/kaggle/working/backups/"
        "objective6_enhancement_protocol_public_v1.1.0.zip"
    )
    archive_hash = deterministic_zip(list(result_root.iterdir()), archive)
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8"
    )

    from huggingface_hub import CommitOperationAdd, HfApi

    hf_api = HfApi(token=hf_token)
    if bool(hf_api.model_info(args.hf_repo, token=hf_token).private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    hf_files = list(result_root.iterdir()) + [archive, archive_checksum]
    hf_commit = hf_api.create_commit(
        repo_id=args.hf_repo, repo_type="model", token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.hf_path.strip('/')}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in hf_files
        ],
        commit_message="protocol: publish Objective 6 v1.1 enhancement lock",
    )

    run_git(["config", "user.name", "Ahmed Zuhair Sabah"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    result_prefix = str(args.result_path).replace("\\", "/").rstrip("/")
    status = run_git(["status", "--porcelain"])
    unexpected = []
    for line in status.splitlines():
        changed = line[3:].strip().split(" -> ")[-1].rstrip("/")
        inside = changed == result_prefix or changed.startswith(f"{result_prefix}/")
        parent = result_prefix.startswith(f"{changed}/")
        if not inside and not parent:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(f"Unexpected Git changes: {unexpected}")
    if status:
        run_git(["add", "--", result_prefix])
        staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
        if not staged or any(not path.startswith(f"{result_prefix}/") for path in staged):
            raise RuntimeError(f"Unexpected staged files: {staged}")
        run_git(["commit", "-m", "protocol: publish Objective 6 v1.1 enhancement"])
    with tempfile.TemporaryDirectory(prefix="git_askpass_") as directory:
        askpass = Path(directory) / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) echo "x-access-token" ;; '
            '*) echo "$GITHUB_TOKEN" ;; esac\n', encoding="utf-8",
        )
        askpass.chmod(0o700)
        environment = dict(os.environ)
        environment["GIT_ASKPASS"] = str(askpass)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        run_git(["push", "origin", "main"], environment)
    github_commit = run_git(["rev-parse", "HEAD"])

    release_endpoint = (
        f"https://api.github.com/repos/{args.github_repo}/releases/tags/"
        f"{args.release_tag}"
    )
    response = github_request("GET", release_endpoint, github_token)
    if response.status_code == 404:
        response = github_request(
            "POST", f"https://api.github.com/repos/{args.github_repo}/releases",
            github_token,
            json={
                "tag_name": args.release_tag,
                "target_commitish": github_commit,
                "name": args.release_title,
                "body": (
                    "Objective 6 v1.1 clinical-guided enhancement protocol, locked "
                    "before implementation, training, and locked-test access."
                ),
                "draft": False, "prerelease": False,
            },
        )
    response.raise_for_status()
    release = response.json()
    existing_assets = {asset["name"] for asset in release.get("assets", [])}
    upload_url = release["upload_url"].split("{")[0]
    for asset in (archive, archive_checksum):
        if asset.name in existing_assets:
            continue
        with asset.open("rb") as stream:
            uploaded = github_request(
                "POST", upload_url, github_token,
                params={"name": asset.name},
                headers={"Content-Type": "application/octet-stream"}, data=stream,
            )
        uploaded.raise_for_status()
    print(json.dumps({
        "protocol_sha256": protocol_hash,
        "final_lock_sha256": lock_hash,
        "public_archive_sha256": archive_hash,
        "hf_commit": getattr(hf_commit, "oid", None),
        "hf_path": (
            f"https://huggingface.co/{args.hf_repo}/tree/main/"
            f"{args.hf_path.strip('/')}"
        ),
        "github_commit": github_commit,
        "github_results": (
            f"https://github.com/{args.github_repo}/tree/main/{result_prefix}"
        ),
        "github_release": release["html_url"],
        "enhancement_rounds_allowed": 1,
        "enhancement_training_started": False,
        "locked_test_evaluated": False,
        "privacy_scan_passed": True,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 6 V1.1 ENHANCEMENT PROTOCOL PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
