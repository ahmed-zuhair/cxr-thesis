#!/usr/bin/env python3
"""Publish a sanitized Objective 2 confirmation protocol lock."""

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
    parser.add_argument("--expected-confirmation-sha256", required=True)
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


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_git(arguments: list[str], *, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def deterministic_zip(files: list[Path], target: Path) -> str:
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as bundle:
        for source in sorted(files, key=lambda item: item.name):
            information = zipfile.ZipInfo(
                source.name,
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o100644 << 16
            bundle.writestr(
                information,
                source.read_bytes(),
                compresslevel=6,
            )
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Protocol archive integrity test failed")
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
    return requests.request(
        method,
        url,
        headers=headers,
        timeout=120,
        **kwargs,
    )


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")
    if not args.protocol_record.is_file() or not args.protocol_checksum.is_file():
        raise FileNotFoundError("Protocol record or checksum is missing")
    actual_protocol_hash = sha256_file(args.protocol_record)
    recorded_protocol_hash = args.protocol_checksum.read_text(
        encoding="utf-8"
    ).split()[0]
    if actual_protocol_hash != args.expected_protocol_sha256:
        raise RuntimeError("Protocol SHA-256 does not match the expected lock")
    if recorded_protocol_hash != actual_protocol_hash:
        raise RuntimeError("Protocol checksum file does not match")
    protocol = json.loads(args.protocol_record.read_text(encoding="utf-8"))
    checks = {
        "confirmation_hash": (
            protocol.get("confirmation_manifest_sha256")
            == args.expected_confirmation_sha256
        ),
        "cases": protocol.get("confirmation_cases") == 5_000,
        "patients": protocol.get("confirmation_patients") == 568,
        "overlap": protocol.get("patient_overlap_with_original_locked_test") == 0,
        "selection_labels": protocol.get("selection_used_labels") is False,
        "selection_predictions": protocol.get("selection_used_predictions") is False,
        "selection_risk": protocol.get("selection_used_risk_scores") is False,
        "label_statistics": (
            protocol.get("confirmation_label_statistics_calculated") is False
        ),
        "status": (
            protocol.get("status")
            == "locked before confirmation-label evaluation"
        ),
        "identifiers": (
            protocol.get("patient_identifiers_included") is False
            and protocol.get("image_identifiers_included") is False
        ),
        "private_manifest": protocol.get("private_manifest_included") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Confirmation protocol checks failed: {checks}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Declared cohort source commit is not in repository history")
    if run_git(["status", "--porcelain"]):
        raise RuntimeError("Repository must be clean before protocol publication")

    result_root = REPOSITORY_ROOT / args.result_path
    result_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "confirmation_protocol_lock_public.json": args.protocol_record,
        "confirmation_protocol_lock_public.json.sha256": args.protocol_checksum,
    }
    for name, source in targets.items():
        target = result_root / name
        if target.is_file() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Existing public protocol file differs: {target}")
        if not target.is_file():
            shutil.copy2(source, target)
    readme = f"""# Objective 2 independent confirmation protocol lock

- Confirmation images: 5,000
- Confirmation patients: 568
- Original locked-test patients excluded: 541
- Patient overlap with original locked test: 0
- Selection seed: 3042
- Confirmation manifest SHA-256: `{args.expected_confirmation_sha256}`
- Public protocol SHA-256: `{actual_protocol_hash}`
- Cohort source commit: `{args.source_commit}`

The confirmation identities were selected from complete official NIH test
patients after excluding every original locked-test patient. Selection used no
disease labels, predictions, or risk scores. This protocol was published before
confirmation-label evaluation. No private manifest, patient identifier, image
identifier, medical image, or case-level prediction is included here.
"""
    readme_path = result_root / "README.md"
    if readme_path.is_file() and readme_path.read_text(encoding="utf-8") != readme:
        raise RuntimeError("Existing protocol README differs")
    readme_path.write_text(readme, encoding="utf-8")
    inventory = {
        "artifact": "Objective 2 confirmation protocol public inventory",
        "confirmation_manifest_sha256": args.expected_confirmation_sha256,
        "protocol_sha256": actual_protocol_hash,
        "private_manifest_included": False,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "medical_images_included": False,
        "confirmation_results_included": False,
        "files": {},
    }
    for path in sorted(result_root.iterdir()):
        if path.is_file() and path.name != "artifact_inventory_public.json":
            inventory["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    inventory_path = result_root / "artifact_inventory_public.json"
    atomic_json(inventory, inventory_path)

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.is_file()
    )
    forbidden = ("patient_id", "image_id", "image_path", "mask_path")
    violations = [item for item in forbidden if f'"{item}"' in serialized]
    if violations:
        raise RuntimeError(f"Public protocol privacy scan failed: {violations}")

    release_root = Path("/kaggle/working/backups")
    release_root.mkdir(parents=True, exist_ok=True)
    archive = release_root / "objective2_confirmation_protocol_public_v1.0.0.zip"
    archive_hash = deterministic_zip(
        [path for path in result_root.iterdir() if path.is_file()],
        archive,
    )
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(
        f"{archive_hash}  {archive.name}\n",
        encoding="utf-8",
    )

    from huggingface_hub import CommitOperationAdd, HfApi

    hf_api = HfApi(token=hf_token)
    hf_info = hf_api.model_info(args.hf_repo, token=hf_token)
    if bool(hf_info.private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    hf_files = [path for path in result_root.iterdir() if path.is_file()]
    hf_files.extend([archive, archive_checksum])
    hf_api.create_commit(
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
        commit_message="protocol: publish independent confirmation lock",
    )

    run_git(["config", "user.name", "Ahmed Zuhair"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    run_git(["add", "--", str(args.result_path).replace("\\", "/")])
    staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
    allowed_prefix = str(args.result_path).replace("\\", "/").rstrip("/") + "/"
    if staged and any(not path.startswith(allowed_prefix) for path in staged):
        raise RuntimeError(f"Unexpected staged files: {staged}")
    if staged:
        run_git(["commit", "-m", "protocol: lock Objective 2 confirmation cohort"])
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

    release_url = (
        f"https://api.github.com/repos/{args.github_repo}/releases/tags/"
        f"{args.release_tag}"
    )
    response = github_request("GET", release_url, github_token)
    if response.status_code == 404:
        response = github_request(
            "POST",
            f"https://api.github.com/repos/{args.github_repo}/releases",
            github_token,
            json={
                "tag_name": args.release_tag,
                "target_commitish": github_commit,
                "name": args.release_title,
                "body": "Label-blind independent confirmation protocol lock.",
                "draft": False,
                "prerelease": False,
            },
        )
    response.raise_for_status()
    release = response.json()
    existing_assets = {item["name"] for item in release.get("assets", [])}
    upload_base = release["upload_url"].split("{")[0]
    for asset in (archive, archive_checksum):
        if asset.name in existing_assets:
            continue
        with asset.open("rb") as handle:
            uploaded = github_request(
                "POST",
                upload_base,
                github_token,
                params={"name": asset.name},
                headers={"Content-Type": "application/octet-stream"},
                data=handle,
            )
        uploaded.raise_for_status()

    print(
        json.dumps(
            {
                "confirmation_manifest_sha256": args.expected_confirmation_sha256,
                "protocol_sha256": actual_protocol_hash,
                "public_archive_sha256": archive_hash,
                "github_commit": github_commit,
                "github_results": (
                    f"https://github.com/{args.github_repo}/tree/main/"
                    f"{str(args.result_path).replace(chr(92), '/')}"
                ),
                "github_release": release["html_url"],
                "hf_path": (
                    f"https://huggingface.co/{args.hf_repo}/tree/main/"
                    f"{args.hf_path.strip('/')}"
                ),
                "confirmation_evaluated": False,
                "private_manifest_published": False,
                "privacy_scan_passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 2 CONFIRMATION PROTOCOL PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
