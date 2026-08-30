#!/usr/bin/env python3
"""Publish the privacy-safe Objective 6 English v2 remediation lock."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publish_objective6_english_v2_protocol import (
    deterministic_zip,
    github_request,
    run_git,
    sha256,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
TRANSLATOR_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remediation-output", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-title", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")

    public = args.remediation_output / "public"
    protocol = public / "objective6_english_v2_remediation_protocol_public.json"
    protocol_checksum = protocol.with_suffix(".json.sha256")
    final_lock = public / "FINAL_OBJECTIVE6_ENGLISH_V2_REMEDIATION_LOCK.json"
    lock_checksum = final_lock.with_suffix(".json.sha256")
    sources = (protocol, protocol_checksum, final_lock, lock_checksum)
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    actual = {"protocol": sha256(protocol), "lock": sha256(final_lock)}
    expected = {
        "protocol": args.expected_protocol_sha256,
        "lock": args.expected_lock_sha256,
    }
    if actual != expected:
        raise RuntimeError(f"Objective 6 English v2 remediation hashes changed: {actual}")
    for checksum, digest in (
        (protocol_checksum, actual["protocol"]),
        (lock_checksum, actual["lock"]),
    ):
        if checksum.read_text(encoding="utf-8").split()[0] != digest:
            raise RuntimeError(f"Checksum mismatch: {checksum}")

    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    lock_payload = json.loads(final_lock.read_text(encoding="utf-8"))
    if (
        protocol_payload.get("artifact")
        != "Objective 6 English v2 factual-translation remediation protocol"
        or protocol_payload.get("version") != "v2.0.1"
        or protocol_payload.get("scientific_scope", {}).get(
            "remediation_attempts_allowed"
        ) != 1
        or protocol_payload.get("scientific_scope", {}).get(
            "candidate_architectures_changed"
        ) is not False
        or protocol_payload.get("enhancement_commitment", {}).get(
            "both_candidates_will_be_developed_if_remediation_passes"
        ) is not True
        or protocol_payload.get("privacy_and_safety", {}).get(
            "enhancement_training_started"
        ) is not False
        or protocol_payload.get("privacy_and_safety", {}).get(
            "locked_test_evaluated"
        ) is not False
        or lock_payload.get("artifact")
        != "Final Objective 6 English v2 factual-remediation lock"
        or lock_payload.get("immutable") is not True
        or lock_payload.get("protocol_sha256") != actual["protocol"]
        or lock_payload.get("translator_revision") != TRANSLATOR_REVISION
        or lock_payload.get("remediation_attempts_allowed") != 1
        or lock_payload.get("remediation_attempts_completed") != 0
        or lock_payload.get("candidate_count") != 2
        or lock_payload.get("enhancement_training_started") is not False
        or lock_payload.get("locked_test_evaluated") is not False
    ):
        raise RuntimeError("Objective 6 English v2 remediation safety state changed")

    result_root = ROOT / args.result_path
    if result_root.exists():
        expected_names = {
            *(path.name for path in sources),
            "README.md",
            "artifact_inventory_public.json",
        }
        existing = {path.name for path in result_root.iterdir() if path.is_file()}
        if existing != expected_names:
            raise RuntimeError("Partial Objective 6 English v2 remediation publication")
    else:
        result_root.mkdir(parents=True)
        for source in sources:
            shutil.copy2(source, result_root / source.name)
        result_root.joinpath("README.md").write_text(
            "# Objective 6 English v2 factual-translation remediation\n\n"
            "This package locks one deterministic preprocessing remediation after "
            "the first English translation failed its preregistered terminology and "
            "measurement gates. The failed result remains unchanged. Candidate "
            "architectures and advancement thresholds are not modified.\n\n"
            "No reports, identifiers, manifests, images, predictions, checkpoints, "
            "case-level diagnostics, or test results are included.\n",
            encoding="utf-8",
        )
        inventory: dict[str, Any] = {
            "artifact": "Objective 6 English v2 remediation public inventory",
            "protocol_sha256": actual["protocol"],
            "final_lock_sha256": actual["lock"],
            "remediation_attempts_allowed": 1,
            "enhancement_candidate_count": 2,
            "private_reports_included": False,
            "translated_reports_included": False,
            "private_manifests_included": False,
            "patient_or_image_identifiers_included": False,
            "medical_images_included": False,
            "case_level_outputs_included": False,
            "private_checkpoints_included": False,
            "locked_test_results_included": False,
            "files": {},
        }
        for path in sorted(result_root.iterdir()):
            inventory["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        write_json(inventory, result_root / "artifact_inventory_public.json")

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.suffix.lower() in {".json", ".md"}
    )
    forbidden = (
        '"patient_id"',
        '"image_id"',
        '"image_path"',
        '"case_code"',
        '"report"',
        '"reference_report"',
        '"generated_report"',
    )
    violations = [value for value in forbidden if value in serialized]
    if violations:
        raise RuntimeError(f"Objective 6 remediation privacy scan failed: {violations}")

    archive = Path(
        "/kaggle/working/backups/"
        "objective6_english_v2_remediation_protocol_public_v2.0.1.zip"
    )
    archive_hash = deterministic_zip(list(result_root.iterdir()), archive)
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8"
    )

    from huggingface_hub import CommitOperationAdd, HfApi

    hf_api = HfApi(token=hf_token)
    if bool(hf_api.model_info(args.hf_repo, token=hf_token).private):
        raise RuntimeError("Public Hugging Face repository is unexpectedly private")
    hf_files = list(result_root.iterdir()) + [archive, archive_checksum]
    hf_commit = hf_api.create_commit(
        repo_id=args.hf_repo,
        repo_type="model",
        token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.hf_path.strip('/')}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in hf_files
        ],
        commit_message="protocol: publish Objective 6 English v2 remediation lock",
    )

    run_git(["config", "user.name", "Ahmed Zuhair Sabah"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    prefix = str(args.result_path).replace("\\", "/").rstrip("/")
    status = run_git(["status", "--porcelain"])
    unexpected = []
    for line in status.splitlines():
        changed = line[3:].strip().split(" -> ")[-1].rstrip("/")
        if not (
            changed == prefix
            or changed.startswith(prefix + "/")
            or prefix.startswith(changed + "/")
        ):
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(f"Unexpected Git changes: {unexpected}")
    if status:
        run_git(["add", "--", prefix])
        run_git(["commit", "-m", "protocol: publish Objective 6 English remediation"])
    with tempfile.TemporaryDirectory(prefix="git_askpass_") as directory:
        askpass = Path(directory) / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) echo "x-access-token" ;; '
            '*) echo "$GITHUB_TOKEN" ;; esac\n',
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        environment = dict(os.environ)
        environment["GIT_ASKPASS"] = str(askpass)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        run_git(["push", "origin", "main"], environment)
    github_commit = run_git(["rev-parse", "HEAD"])

    endpoint = (
        f"https://api.github.com/repos/{args.github_repo}/releases/tags/"
        f"{args.release_tag}"
    )
    response = github_request("GET", endpoint, github_token)
    if response.status_code == 404:
        response = github_request(
            "POST",
            f"https://api.github.com/repos/{args.github_repo}/releases",
            github_token,
            json={
                "tag_name": args.release_tag,
                "target_commitish": github_commit,
                "name": args.release_title,
                "body": (
                    "Privacy-safe Objective 6 English v2 factual-remediation lock "
                    "published before correction or enhancement training."
                ),
                "draft": False,
                "prerelease": False,
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
                "POST",
                upload_url,
                github_token,
                params={"name": asset.name},
                headers={"Content-Type": "application/octet-stream"},
                data=stream,
            )
        uploaded.raise_for_status()

    print(
        json.dumps(
            {
                "protocol_sha256": actual["protocol"],
                "final_lock_sha256": actual["lock"],
                "public_archive_sha256": archive_hash,
                "hf_commit": getattr(hf_commit, "oid", None),
                "hf_path": (
                    f"https://huggingface.co/{args.hf_repo}/tree/main/"
                    f"{args.hf_path.strip('/')}"
                ),
                "github_commit": github_commit,
                "github_results": (
                    f"https://github.com/{args.github_repo}/tree/main/{prefix}"
                ),
                "github_release": release["html_url"],
                "remediation_performed": False,
                "enhancement_training_started": False,
                "locked_test_evaluated": False,
                "privacy_scan_passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 6 ENGLISH V2 REMEDIATION PROTOCOL PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
