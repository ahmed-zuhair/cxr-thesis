#!/usr/bin/env python3
"""Publish the sanitized Objective 4 XAI protocol before explanations."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-record", type=Path, required=True)
    parser.add_argument("--protocol-checksum", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-private-cohort-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-title", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(
    protocol: dict[str, object],
    *,
    expected_private_hash: str,
    expected_checkpoint_hash: str,
) -> dict[str, bool]:
    cohort = protocol.get("cohort", {})
    protections = protocol.get("protections", {})
    if not isinstance(cohort, dict) or not isinstance(protections, dict):
        raise RuntimeError("Malformed Objective 4 protocol")
    label_counts = cohort.get("target_label_counts", {})
    checks = {
        "artifact": protocol.get("artifact")
        == "Objective 4 quantitative XAI protocol lock",
        "status": protocol.get("status")
        == "locked_before_explanation_generation",
        "objective": protocol.get("objective") == 4,
        "model": protocol.get("model") == "densenet121",
        "checkpoint": protocol.get("expected_checkpoint_sha256")
        == expected_checkpoint_hash,
        "private_hash": protocol.get("private_xai_cohort_sha256")
        == expected_private_hash,
        "split": cohort.get("split") == "val",
        "cases": cohort.get("cases") == 240,
        "patients": cohort.get("unique_patients") == 240,
        "images": cohort.get("unique_images") == 240,
        "per_label": cohort.get("cases_per_target_label") == 20,
        "label_count": isinstance(label_counts, dict) and len(label_counts) == 12,
        "balanced": isinstance(label_counts, dict)
        and set(label_counts.values()) == {20},
        "predictions": cohort.get("predictions_used_for_selection") is False,
        "risk": cohort.get("risk_scores_used_for_selection") is False,
        "test_manifest": protections.get("test_manifest_opened") is False,
        "test_labels": protections.get("test_labels_accessed") is False,
        "test_evaluation": protections.get("test_evaluated") is False,
        "manual_masking": protections.get("manual_masking_required") is False,
        "private_manifest": protections.get(
            "private_manifest_allowed_for_public_upload"
        )
        is False,
        "medical_images": protections.get("medical_images_public") is False,
        "case_explanations": protections.get("case_level_explanations_public")
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Objective 4 protocol checks failed: {checks}")
    return checks


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_git(arguments: list[str], *, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=REPOSITORY_ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def deterministic_zip(files: list[Path], target: Path) -> str:
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for source in sorted(files, key=lambda item: item.name):
            information = zipfile.ZipInfo(source.name, (1980, 1, 1, 0, 0, 0))
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o100644 << 16
            bundle.writestr(information, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Objective 4 public archive integrity test failed")
    return sha256_file(target)


def github_request(method: str, url: str, token: str, **kwargs):
    import requests

    headers = dict(kwargs.pop("headers", {}))
    headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return requests.request(method, url, headers=headers, timeout=120, **kwargs)


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")
    if not args.protocol_record.is_file() or not args.protocol_checksum.is_file():
        raise FileNotFoundError("Objective 4 public protocol or checksum is missing")

    protocol_hash = sha256_file(args.protocol_record)
    recorded_hash = args.protocol_checksum.read_text(encoding="utf-8").split()[0]
    if protocol_hash != args.expected_protocol_sha256:
        raise RuntimeError("Objective 4 public protocol SHA-256 is unexpected")
    if recorded_hash != protocol_hash:
        raise RuntimeError("Objective 4 protocol checksum file does not match")
    protocol = json.loads(args.protocol_record.read_text(encoding="utf-8"))
    checks = validate_protocol(
        protocol,
        expected_private_hash=args.expected_private_cohort_sha256,
        expected_checkpoint_hash=args.expected_checkpoint_sha256,
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"],
        cwd=REPOSITORY_ROOT, check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Objective 4 protocol source commit is not in history")
    if run_git(["status", "--porcelain"]):
        raise RuntimeError("Repository must be clean before publication")

    result_root = REPOSITORY_ROOT / args.result_path
    if result_root.exists():
        raise FileExistsError(f"Public Objective 4 result already exists: {result_root}")
    result_root.mkdir(parents=True)
    shutil.copy2(
        args.protocol_record,
        result_root / "objective4_xai_protocol_lock_public.json",
    )
    shutil.copy2(
        args.protocol_checksum,
        result_root / "objective4_xai_protocol_lock_public.sha256",
    )
    readme = f"""# Objective 4 quantitative XAI protocol lock

- Model: independently confirmed DenseNet-121
- Cohort: 240 validation cases from 240 unique patients
- Allocation: 20 label-positive cases for each of 12 disease targets
- Methods: Grad-CAM and Integrated Gradients
- Public protocol SHA-256: `{protocol_hash}`
- Private cohort SHA-256: `{args.expected_private_cohort_sha256}`
- DenseNet checkpoint SHA-256: `{args.expected_checkpoint_sha256}`
- Protocol source commit: `{args.source_commit}`

The cohort was selected deterministically before explanation generation.
Predictions and risk scores were not used for selection. The locked test
manifest was not opened or evaluated. The private cohort manifest, patient and
image identifiers, medical images, and case-level explanation maps are not
included in this public artifact.
"""
    (result_root / "README.md").write_text(readme, encoding="utf-8")
    inventory = {
        "artifact": "Objective 4 XAI protocol public inventory",
        "protocol_sha256": protocol_hash,
        "private_cohort_sha256": args.expected_private_cohort_sha256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "protocol_checks": checks,
        "private_manifest_included": False,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "medical_images_included": False,
        "case_level_explanations_included": False,
        "explanation_results_included": False,
        "files": {},
    }
    for path in sorted(result_root.iterdir()):
        inventory["files"][path.name] = {
            "bytes": path.stat().st_size, "sha256": sha256_file(path)
        }
    atomic_json(inventory, result_root / "artifact_inventory_public.json")

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.is_file()
    )
    forbidden = (
        '"patient_id"', '"image_id"', '"image_path"', '"mask_path"',
        "ADAPT-", "NIH-CAND-",
    )
    violations = [item for item in forbidden if item in serialized]
    if violations:
        raise RuntimeError(f"Objective 4 public privacy scan failed: {violations}")

    backup_root = Path("/kaggle/working/backups")
    backup_root.mkdir(parents=True, exist_ok=True)
    archive = backup_root / "objective4_xai_protocol_public_v1.0.0.zip"
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
        commit_message="protocol: publish Objective 4 XAI lock",
    )

    run_git(["config", "user.name", "Ahmed Zuhair"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    normalized_result = str(args.result_path).replace("\\", "/")
    run_git(["add", "--", normalized_result])
    staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
    prefix = normalized_result.rstrip("/") + "/"
    if not staged or any(not path.startswith(prefix) for path in staged):
        raise RuntimeError(f"Unexpected Objective 4 staged files: {staged}")
    run_git(["commit", "-m", "protocol: publish Objective 4 XAI lock"])
    with tempfile.TemporaryDirectory(prefix="git_askpass_") as directory:
        askpass = Path(directory) / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) echo "x-access-token" ;; *) echo "$GITHUB_TOKEN" ;; esac\n',
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        environment = dict(os.environ)
        environment["GIT_ASKPASS"] = str(askpass)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        run_git(["push", "origin", "main"], environment=environment)
    github_commit = run_git(["rev-parse", "HEAD"])

    release_api = (
        f"https://api.github.com/repos/{args.github_repo}/releases/tags/"
        f"{args.release_tag}"
    )
    response = github_request("GET", release_api, github_token)
    if response.status_code == 404:
        response = github_request(
            "POST",
            f"https://api.github.com/repos/{args.github_repo}/releases",
            github_token,
            json={
                "tag_name": args.release_tag,
                "target_commitish": github_commit,
                "name": args.release_title,
                "body": "Validation-only Objective 4 quantitative XAI protocol lock.",
                "draft": False, "prerelease": False,
            },
        )
    response.raise_for_status()
    release = response.json()
    existing = {asset["name"] for asset in release.get("assets", [])}
    upload_url = release["upload_url"].split("{")[0]
    for asset in (archive, archive_checksum):
        if asset.name in existing:
            continue
        with asset.open("rb") as handle:
            uploaded = github_request(
                "POST", upload_url, github_token,
                params={"name": asset.name},
                headers={"Content-Type": "application/octet-stream"},
                data=handle,
            )
        uploaded.raise_for_status()

    print(json.dumps({
        "protocol_sha256": protocol_hash,
        "private_cohort_sha256": args.expected_private_cohort_sha256,
        "public_archive_sha256": archive_hash,
        "hf_commit": getattr(hf_commit, "oid", None),
        "hf_path": f"https://huggingface.co/{args.hf_repo}/tree/main/{args.hf_path.strip('/')}",
        "github_commit": github_commit,
        "github_results": f"https://github.com/{args.github_repo}/tree/main/{normalized_result}",
        "github_release": release["html_url"],
        "test_evaluated": False,
        "private_manifest_published": False,
        "medical_images_published": False,
        "case_level_explanations_published": False,
        "privacy_scan_passed": True,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 4 XAI PROTOCOL PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
