#!/usr/bin/env python3
"""Publish sanitized aggregate Objective 2 locked-test results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-output", type=Path, required=True)
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


def run_git(arguments: list[str], *, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def deterministic_zip(files: list[Path], target: Path) -> str:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted(files, key=lambda item: item.name):
            information = zipfile.ZipInfo(source.name, date_time=(1980, 1, 1, 0, 0, 0))
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o100644 << 16
            bundle.writestr(information, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Locked-test public archive integrity failed")
    return sha256_file(target)


def privacy_scan(root: Path) -> None:
    forbidden_extensions = {".pt", ".npz", ".npy", ".dcm", ".nii"}
    identifier_pattern = re.compile(r"\b\d{8}_\d{3}\.png\b|\bnih-\d{8}", re.I)
    forbidden_keys = {"patient_id", "image_id", "image_path", "mask_path", "filename"}
    violations: list[str] = []

    def keys(value) -> set[str]:
        if isinstance(value, dict):
            nested = set(map(str, value))
            for item in value.values():
                nested.update(keys(item))
            return nested
        if isinstance(value, list):
            nested: set[str] = set()
            for item in value:
                nested.update(keys(item))
            return nested
        return set()

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in forbidden_extensions:
            violations.append(f"forbidden binary: {path.name}")
        if path.suffix.lower() in {".json", ".md", ".csv", ".sha256"}:
            text = path.read_text(encoding="utf-8")
            if identifier_pattern.search(text):
                violations.append(f"identifier pattern: {path.name}")
            if path.suffix.lower() == ".json":
                overlap = keys(json.loads(text)) & forbidden_keys
                if overlap:
                    violations.append(f"private JSON keys in {path.name}: {sorted(overlap)}")
    if violations:
        raise RuntimeError("Privacy scan failed:\n" + "\n".join(violations))


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
    public_source = args.evaluation_output / "public"
    final_lock_path = args.evaluation_output / "FINAL_LOCKED_TEST_EVALUATION.json"
    summary_source = public_source / "objective2_locked_test_summary_public.json"
    figure_source = public_source / "objective2_locked_test_model_comparison.png"
    for path in (final_lock_path, summary_source, figure_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = json.loads(final_lock_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_source.read_text(encoding="utf-8"))
    if lock.get("test_evaluated") is not True or summary.get("test_evaluated") is not True:
        raise RuntimeError("Locked-test evaluation is not finalized")
    if summary.get("test_used_for_model_selection") is not False:
        raise RuntimeError("Test cohort was used for model selection")
    if int(summary.get("test_evaluation_count_per_model", 0)) != 1:
        raise RuntimeError("Unexpected locked-test evaluation count")
    if sha256_file(summary_source) != lock.get("summary_sha256"):
        raise RuntimeError("Locked-test summary hash does not match final lock")
    if sha256_file(figure_source) != lock.get("figure_sha256"):
        raise RuntimeError("Locked-test figure hash does not match final lock")
    if set(summary.get("models", {})) != {"cnn", "attention_cnn", "vit", "gcn", "gat"}:
        raise RuntimeError("Locked-test summary does not contain all five models")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
    ).returncode != 0:
        raise RuntimeError("Declared source commit is not in current history")
    result_root = REPOSITORY_ROOT / args.result_path
    result_prefix = str(args.result_path).replace("\\", "/").rstrip("/") + "/"
    status_lines = run_git(["status", "--porcelain"]).splitlines()
    unexpected_changes = []
    for line in status_lines:
        changed_path = line[3:].replace("\\", "/")
        if changed_path.rstrip("/") == result_prefix.rstrip("/"):
            continue
        if not changed_path.startswith(result_prefix):
            unexpected_changes.append(line)
    if unexpected_changes:
        raise RuntimeError(f"Unexpected repository changes: {unexpected_changes}")

    result_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_source, result_root / summary_source.name)
    shutil.copy2(figure_source, result_root / figure_source.name)
    public_lock = {
        key: lock[key]
        for key in (
            "artifact",
            "test_manifest_sha256",
            "checkpoint_sha256",
            "summary_sha256",
            "figure_sha256",
            "completed_models",
            "validation_thresholds_reused_without_change",
            "test_used_for_model_selection",
            "test_evaluated",
        )
    }
    (result_root / "locked_test_evaluation_lock_public.json").write_text(
        json.dumps(public_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (result_root / "README.md").write_text(
        "# Objective 2 Final Locked-Test Comparison\n\n"
        "This directory contains the aggregate final comparison of the frozen CNN, "
        "Attention-CNN, compact ViT, GCN and GAT candidates. Validation-selected "
        "thresholds were reused without change. The test cohort was evaluated once "
        "per model and was not used for training, threshold tuning or model selection.\n\n"
        "No patient/image identifiers, medical images, private manifests or case-level "
        "predictions are included.\n",
        encoding="utf-8",
    )
    inventory = {
        "artifact": "Objective 2 locked-test public artifact inventory",
        "files": {},
        "case_level_predictions_included": False,
        "private_data_included": False,
    }
    inventory_name = "artifact_inventory_public.json"
    for path in sorted(result_root.iterdir()):
        if path.is_file() and path.name != inventory_name:
            inventory["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    (result_root / inventory_name).write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expected_result_files = {
        "README.md",
        "artifact_inventory_public.json",
        "locked_test_evaluation_lock_public.json",
        summary_source.name,
        figure_source.name,
    }
    actual_result_files = {
        path.name for path in result_root.iterdir() if path.is_file()
    }
    if actual_result_files != expected_result_files:
        raise RuntimeError(
            "Unexpected public result files: "
            f"{sorted(actual_result_files - expected_result_files)}"
        )
    privacy_scan(result_root)

    backups = Path("/kaggle/working/backups")
    backups.mkdir(parents=True, exist_ok=True)
    archive = backups / "objective2_locked_test_comparison_public_v1.0.0.zip"
    archive_hash = deterministic_zip(
        [path for path in result_root.iterdir() if path.is_file()], archive
    )
    checksum = archive.with_suffix(".zip.sha256")
    checksum.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    hf_api = HfApi(token=hf_token)
    if bool(hf_api.model_info(args.hf_repo, token=hf_token).private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    hf_files = [path for path in result_root.iterdir() if path.is_file()] + [archive, checksum]
    remote_files = set(
        hf_api.list_repo_files(args.hf_repo, repo_type="model", token=hf_token)
    )
    hf_cache = backups / "objective2_locked_test_hf_verification"
    hf_cache.mkdir(parents=True, exist_ok=True)
    operations = []
    for path in hf_files:
        remote_path = f"{args.hf_path.strip('/')}/{path.name}"
        if remote_path in remote_files:
            downloaded = Path(
                hf_hub_download(
                    args.hf_repo,
                    filename=remote_path,
                    repo_type="model",
                    token=hf_token,
                    local_dir=hf_cache,
                    force_download=True,
                )
            )
            if sha256_file(downloaded) != sha256_file(path):
                raise RuntimeError(f"Existing HF artifact differs: {remote_path}")
            continue
        operations.append(
            CommitOperationAdd(
                path_in_repo=remote_path,
                path_or_fileobj=str(path),
            )
        )
    if operations:
        hf_api.create_commit(
            repo_id=args.hf_repo,
            repo_type="model",
            token=hf_token,
            operations=operations,
            commit_message="results: publish Objective 2 locked-test comparison",
        )

    run_git(["config", "user.name", "Ahmed Zuhair"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    run_git(["add", "--", str(args.result_path).replace("\\", "/")])
    staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
    if any(not path.startswith(result_prefix) for path in staged):
        raise RuntimeError(f"Unexpected staged files: {staged}")
    if staged:
        run_git(["commit", "-m", "results: publish Objective 2 locked-test comparison"])
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

    release_response = github_request(
        "GET",
        f"https://api.github.com/repos/{args.github_repo}/releases/tags/{args.release_tag}",
        github_token,
    )
    if release_response.status_code == 404:
        release_response = github_request(
            "POST",
            f"https://api.github.com/repos/{args.github_repo}/releases",
            github_token,
            json={
                "tag_name": args.release_tag,
                "target_commitish": github_commit,
                "name": args.release_title,
                "body": "Final aggregate, test-once Objective 2 model comparison.",
                "draft": False,
                "prerelease": False,
            },
        )
    release_response.raise_for_status()
    release = release_response.json()
    existing = {asset["name"] for asset in release.get("assets", [])}
    upload_url = release["upload_url"].split("{")[0]
    for asset in (archive, checksum):
        if asset.name in existing:
            continue
        with asset.open("rb") as handle:
            response = github_request(
                "POST",
                upload_url,
                github_token,
                params={"name": asset.name},
                headers={"Content-Type": "application/octet-stream"},
                data=handle,
            )
        response.raise_for_status()
    print(
        json.dumps(
            {
                "summary_sha256": lock["summary_sha256"],
                "figure_sha256": lock["figure_sha256"],
                "archive_sha256": archive_hash,
                "hf_path": f"https://huggingface.co/{args.hf_repo}/tree/main/{args.hf_path.strip('/')}",
                "github_results": f"https://github.com/{args.github_repo}/tree/main/{str(args.result_path).replace(chr(92), '/')}",
                "github_commit": github_commit,
                "github_release": release["html_url"],
                "privacy_scan_passed": True,
                "test_used_for_model_selection": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 2 LOCKED-TEST COMPARISON PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
