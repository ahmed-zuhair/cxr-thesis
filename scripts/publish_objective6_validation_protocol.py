#!/usr/bin/env python3
"""Publish the Objective 6 pre-generation validation-evaluation lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_git(arguments: list[str], environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def github_request(method: str, url: str, token: str, **kwargs):
    import requests

    headers = dict(kwargs.pop("headers", {}))
    headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return requests.request(method, url, headers=headers, timeout=120, **kwargs)


def deterministic_zip(files: list[Path], target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for source in sorted(files, key=lambda item: item.name):
            information = zipfile.ZipInfo(source.name, (1980, 1, 1, 0, 0, 0))
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o100644 << 16
            archive.writestr(information, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Objective 6 validation-protocol archive is corrupt")
    return sha256(target)


def validate(protocol: dict[str, Any], lock: dict[str, Any]) -> None:
    if protocol.get("artifact") != (
        "Objective 6 validation generation and comparison protocol"
    ):
        raise RuntimeError("Unexpected Objective 6 validation protocol")
    if protocol.get("status") != (
        "locked after validation-loss training and before full validation "
        "generation or locked-test access"
    ):
        raise RuntimeError("Objective 6 validation protocol status changed")
    primary = protocol.get("primary_system")
    if not isinstance(primary, dict) or (
        primary.get("variant") != "multimodal"
        or primary.get("selected_after_generation_metrics") is not False
    ):
        raise RuntimeError("Objective 6 primary model is not preregistered")
    safety = protocol.get("safety_state")
    if not isinstance(safety, dict) or any(
        safety.get(field) is not False
        for field in (
            "full_validation_generation_started", "locked_test_manifest_opened",
            "locked_test_reports_accessed", "locked_test_evaluated",
        )
    ):
        raise RuntimeError("Objective 6 protocol is not pre-generation/test-blind")
    candidates = protocol.get("candidate_training_results")
    if not isinstance(candidates, dict) or set(candidates) != {
        "image_only", "multimodal"
    }:
        raise RuntimeError("Objective 6 candidate set changed")
    for variant, record in candidates.items():
        if not isinstance(record, dict) or (
            record.get("variant") != variant
            or record.get("test_evaluated") is not False
            or not isinstance(record.get("checkpoint_sha256"), str)
        ):
            raise RuntimeError(f"Invalid Objective 6 candidate: {variant}")
    if lock.get("artifact") != "Final Objective 6 pre-validation-generation lock":
        raise RuntimeError("Unexpected Objective 6 final lock")
    if (
        lock.get("immutable") is not True
        or lock.get("primary_system") != "multimodal"
        or lock.get("full_validation_generation_started") is not False
        or lock.get("locked_test_evaluated") is not False
        or lock.get("locked_test_evaluation_count") != 0
    ):
        raise RuntimeError("Objective 6 final lock is not immutable/test-blind")


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")

    protocol_source = (
        args.lock_directory / "objective6_validation_evaluation_protocol_public.json"
    )
    protocol_checksum = protocol_source.with_suffix(".json.sha256")
    lock_source = (
        args.lock_directory / "FINAL_OBJECTIVE6_VALIDATION_EVALUATION_LOCK.json"
    )
    lock_checksum = lock_source.with_suffix(".json.sha256")
    sources = (protocol_source, protocol_checksum, lock_source, lock_checksum)
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol_hash = sha256(protocol_source)
    lock_hash = sha256(lock_source)
    if protocol_hash != args.expected_protocol_sha256:
        raise RuntimeError("Objective 6 validation protocol hash changed")
    if lock_hash != args.expected_lock_sha256:
        raise RuntimeError("Objective 6 validation final-lock hash changed")
    if protocol_checksum.read_text(encoding="utf-8").split()[0] != protocol_hash:
        raise RuntimeError("Objective 6 validation protocol checksum mismatch")
    if lock_checksum.read_text(encoding="utf-8").split()[0] != lock_hash:
        raise RuntimeError("Objective 6 validation lock checksum mismatch")
    protocol = json.loads(protocol_source.read_text(encoding="utf-8"))
    lock = json.loads(lock_source.read_text(encoding="utf-8"))
    validate(protocol, lock)
    if lock.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Final lock does not identify the validation protocol")

    result_root = ROOT / args.result_path
    expected_names = {
        *(path.name for path in sources),
        "README.md", "artifact_inventory_public.json",
    }
    if result_root.exists():
        existing = {path.name for path in result_root.iterdir() if path.is_file()}
        if existing != expected_names:
            raise RuntimeError("Partial Objective 6 validation publication exists")
    else:
        result_root.mkdir(parents=True)
        for source in sources:
            shutil.copy2(source, result_root / source.name)
        (result_root / "README.md").write_text(
            "# Objective 6 validation-generation and comparison protocol\n\n"
            "This immutable protocol was locked after test-blind training and "
            "before full validation report generation. The multimodal system "
            "remains the preregistered primary system; image-only and frozen "
            "nearest-image retrieval are mandatory comparators. No reports, "
            "identifiers, medical images, private manifests, checkpoints, or "
            "case-level results are included. The locked test remains unopened.\n",
            encoding="utf-8",
        )
        inventory: dict[str, Any] = {
            "artifact": "Objective 6 validation-protocol public inventory",
            "protocol_sha256": protocol_hash,
            "final_lock_sha256": lock_hash,
            "private_manifests_included": False,
            "patient_or_image_identifiers_included": False,
            "medical_images_included": False,
            "raw_or_generated_reports_included": False,
            "case_level_outputs_included": False,
            "private_checkpoints_included": False,
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
        '"report"', '"report_normalised"', '"labels_by_sentence"',
    )
    violations = [value for value in forbidden if value in serialized]
    if violations:
        raise RuntimeError(f"Objective 6 public privacy scan failed: {violations}")

    archive = Path(
        "/kaggle/working/backups/"
        "objective6_validation_evaluation_protocol_public_v1.0.0.zip"
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
        commit_message="protocol: publish Objective 6 validation evaluation lock",
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
        run_git([
            "commit", "-m",
            "protocol: publish Objective 6 validation evaluation lock",
        ])
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
                    "Objective 6 validation-generation and comparison protocol, "
                    "published before full validation generation and any locked-test access."
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
        "full_validation_generation_started": False,
        "locked_test_evaluated": False,
        "private_manifests_published": False,
        "raw_or_generated_reports_published": False,
        "privacy_scan_passed": True,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 6 VALIDATION-EVALUATION PROTOCOL PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
