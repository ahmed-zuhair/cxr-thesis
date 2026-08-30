#!/usr/bin/env python3
"""Publish the privacy-safe Objective 6 English v2 pre-translation protocol."""

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
EXPECTED_TRANSLATOR_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-output", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-cohort-summary-sha256", required=True)
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


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def deterministic_zip(files: list[Path], target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in sorted(files, key=lambda item: item.name):
            info = zipfile.ZipInfo(source.name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Objective 6 English v2 protocol archive is corrupt")
    return sha256(target)


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


def github_request(method: str, url: str, token: str, **kwargs: Any) -> Any:
    import requests

    headers = dict(kwargs.pop("headers", {}))
    headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return requests.request(method, url, headers=headers, timeout=120, **kwargs)


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")

    public = args.protocol_output / "public"
    protocol = public / "objective6_english_v2_protocol_public.json"
    protocol_checksum = protocol.with_suffix(".json.sha256")
    cohort = public / "objective6_english_v2_cohort_summary_public.json"
    cohort_checksum = cohort.with_suffix(".json.sha256")
    final_lock = public / "FINAL_OBJECTIVE6_ENGLISH_V2_PROTOCOL_LOCK.json"
    lock_checksum = final_lock.with_suffix(".json.sha256")
    sources = (
        protocol, protocol_checksum, cohort, cohort_checksum, final_lock, lock_checksum
    )
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    actual = {
        "protocol": sha256(protocol),
        "cohort": sha256(cohort),
        "lock": sha256(final_lock),
    }
    expected = {
        "protocol": args.expected_protocol_sha256,
        "cohort": args.expected_cohort_summary_sha256,
        "lock": args.expected_lock_sha256,
    }
    if actual != expected:
        raise RuntimeError(f"Objective 6 English v2 public hashes changed: {actual}")
    for path, digest in (
        (protocol_checksum, actual["protocol"]),
        (cohort_checksum, actual["cohort"]),
        (lock_checksum, actual["lock"]),
    ):
        if path.read_text(encoding="utf-8").split()[0] != digest:
            raise RuntimeError(f"Checksum mismatch: {path}")

    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    cohort_payload = json.loads(cohort.read_text(encoding="utf-8"))
    lock_payload = json.loads(final_lock.read_text(encoding="utf-8"))
    if (
        protocol_payload.get("artifact")
        != "Objective 6 English fact-aware report-generation v2 protocol"
        or protocol_payload.get("version") != "v2.0.0"
        or protocol_payload.get("v2_candidates", {}).get("candidate_count") != 2
        or protocol_payload.get("english_reference_pipeline", {}).get(
            "immutable_model_revision"
        ) != EXPECTED_TRANSLATOR_REVISION
        or protocol_payload.get("privacy_and_safety", {}).get(
            "locked_test_evaluated"
        ) is not False
        or cohort_payload.get("translation_performed") is not False
        or cohort_payload.get("training_performed") is not False
        or cohort_payload.get("patient_overlap") != 0
        or lock_payload.get("immutable") is not True
        or lock_payload.get("protocol_sha256") != actual["protocol"]
        or lock_payload.get("cohort_summary_sha256") != actual["cohort"]
        or lock_payload.get("translator_revision") != EXPECTED_TRANSLATOR_REVISION
        or lock_payload.get("candidate_count") != 2
        or lock_payload.get("translation_performed") is not False
        or lock_payload.get("v2_training_started") is not False
        or lock_payload.get("original_validation_opened") is not False
        or lock_payload.get("locked_test_manifest_opened") is not False
        or lock_payload.get("locked_test_reports_accessed") is not False
        or lock_payload.get("locked_test_evaluated") is not False
    ):
        raise RuntimeError("Objective 6 English v2 protocol safety state changed")

    result_root = ROOT / args.result_path
    if result_root.exists():
        expected_names = {
            *(path.name for path in sources),
            "README.md", "artifact_inventory_public.json",
        }
        existing = {path.name for path in result_root.iterdir() if path.is_file()}
        if existing != expected_names:
            raise RuntimeError("Partial Objective 6 English v2 publication exists")
    else:
        result_root.mkdir(parents=True)
        for source in sources:
            shutil.copy2(source, result_root / source.name)
        result_root.joinpath("README.md").write_text(
            "# Objective 6 English fact-aware report generation v2\n\n"
            "This package preregisters the separate English v2 remedial extension "
            "after the published negative v1.1 validation decision. It freezes the "
            "private patient-disjoint development boundary, local NLLB translation "
            "revision, two fact-aware candidates, advancement thresholds, privacy "
            "rules, and the one-evaluation policy before translation or training.\n\n"
            "No reports, identifiers, images, manifests, predictions, checkpoints, "
            "translations, or test results are included.\n",
            encoding="utf-8",
        )
        inventory: dict[str, Any] = {
            "artifact": "Objective 6 English v2 public protocol inventory",
            "protocol_sha256": actual["protocol"],
            "cohort_summary_sha256": actual["cohort"],
            "final_lock_sha256": actual["lock"],
            "translator_revision": EXPECTED_TRANSLATOR_REVISION,
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
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
        write_json(inventory, result_root / "artifact_inventory_public.json")

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.suffix.lower() in {".json", ".md"}
    )
    forbidden = (
        '"patient_id"', '"image_id"', '"image_path"', '"case_code"',
        '"report"', '"reference_report"', '"generated_report"',
    )
    violations = [value for value in forbidden if value in serialized]
    if violations:
        raise RuntimeError(f"Objective 6 English v2 privacy scan failed: {violations}")

    archive = Path(
        "/kaggle/working/backups/"
        "objective6_english_v2_protocol_public_v2.0.0.zip"
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
        repo_id=args.hf_repo, repo_type="model", token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.hf_path.strip('/')}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in hf_files
        ],
        commit_message="protocol: publish Objective 6 English v2 pre-translation lock",
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
        run_git(["commit", "-m", "protocol: publish Objective 6 English v2 lock"])
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

    endpoint = (
        f"https://api.github.com/repos/{args.github_repo}/releases/tags/"
        f"{args.release_tag}"
    )
    response = github_request("GET", endpoint, github_token)
    if response.status_code == 404:
        response = github_request(
            "POST", f"https://api.github.com/repos/{args.github_repo}/releases",
            github_token,
            json={
                "tag_name": args.release_tag,
                "target_commitish": github_commit,
                "name": args.release_title,
                "body": (
                    "Privacy-safe Objective 6 English v2 protocol locked before "
                    "translation, model training, generation, or test access."
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
        "protocol_sha256": actual["protocol"],
        "cohort_summary_sha256": actual["cohort"],
        "final_lock_sha256": actual["lock"],
        "translator_revision": EXPECTED_TRANSLATOR_REVISION,
        "public_archive_sha256": archive_hash,
        "hf_commit": getattr(hf_commit, "oid", None),
        "hf_path": f"https://huggingface.co/{args.hf_repo}/tree/main/{args.hf_path.strip('/')}",
        "github_commit": github_commit,
        "github_results": f"https://github.com/{args.github_repo}/tree/main/{prefix}",
        "github_release": release["html_url"],
        "translation_performed": False,
        "training_performed": False,
        "locked_test_evaluated": False,
        "privacy_scan_passed": True,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 6 ENGLISH V2 PRE-TRANSLATION PROTOCOL PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
