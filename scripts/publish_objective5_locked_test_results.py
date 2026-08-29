#!/usr/bin/env python3
"""Publish sanitized aggregate Objective 5 locked-test results."""

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

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_NAME = "objective5_external_locked_test_summary_public.json"
FIGURE_NAME = "objective5_external_locked_test_metrics.png"
FINAL_LOCK_NAME = "FINAL_OBJECTIVE5_LOCKED_TEST_EVALUATION.json"
EXPECTED_DATASETS = {"chexpert", "padchest"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-title", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_git(arguments: list[str], environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
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


def deterministic_zip(files: list[Path], target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in sorted(files, key=lambda item: item.name):
            information = zipfile.ZipInfo(source.name, (1980, 1, 1, 0, 0, 0))
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o100644 << 16
            archive.writestr(information, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Objective 5 final public archive is corrupt")
    return sha256_file(target)


def validate(summary: dict[str, object], lock: dict[str, object]) -> None:
    if summary.get("artifact") != "Objective 5 final external-domain locked-test evaluation":
        raise RuntimeError("Unexpected Objective 5 final summary")
    if set(summary.get("datasets", {})) != EXPECTED_DATASETS:
        raise RuntimeError("Objective 5 final datasets changed")
    required_true = (
        "temperatures_reused_without_change",
        "thresholds_reused_without_change",
        "test_evaluated",
    )
    required_false = (
        "test_threshold_tuning",
        "test_temperature_tuning",
        "test_used_for_model_selection",
        "patient_identifiers_published",
        "image_identifiers_published",
        "medical_images_published",
        "case_level_predictions_published",
        "private_manifests_published",
    )
    if any(summary.get(field) is not True for field in required_true):
        raise RuntimeError("Objective 5 frozen-evaluation declarations are invalid")
    if any(summary.get(field) is not False for field in required_false):
        raise RuntimeError("Objective 5 privacy/test-use declarations are invalid")
    if summary.get("test_evaluation_count_per_dataset") != 1:
        raise RuntimeError("Objective 5 locked test was not evaluated exactly once")
    for dataset, result in summary["datasets"].items():
        if result.get("selected_candidate") != "adapted DenseNet-121":
            raise RuntimeError(f"Unexpected selected candidate for {dataset}")
        if result.get("test_cases") not in {200, 5_000}:
            raise RuntimeError(f"Unexpected test count for {dataset}")
        if result.get("test_patients") != result.get("test_cases"):
            raise RuntimeError(f"{dataset} is not one image per test patient")
        macro = result.get("test_metrics", {}).get("macro", {})
        if not all(name in macro for name in ("auroc", "auprc", "f1", "brier", "ece")):
            raise RuntimeError(f"{dataset} metrics are incomplete")
    if lock.get("immutable") is not True or lock.get("test_evaluated") is not True:
        raise RuntimeError("Objective 5 final lock is not immutable/finalized")
    if lock.get("test_evaluation_count_per_dataset") != 1:
        raise RuntimeError("Objective 5 final lock has an invalid evaluation count")
    if lock.get("completed_datasets") != ["chexpert", "padchest"]:
        raise RuntimeError("Objective 5 final lock is incomplete")


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")
    public_root = args.evaluation_output / "public"
    summary_source = public_root / SUMMARY_NAME
    summary_checksum = public_root / f"{SUMMARY_NAME}.sha256"
    figure_source = public_root / FIGURE_NAME
    figure_checksum = public_root / f"{FIGURE_NAME}.sha256"
    lock_source = args.evaluation_output / FINAL_LOCK_NAME
    for path in (summary_source, summary_checksum, figure_source, figure_checksum, lock_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary_hash = sha256_file(summary_source)
    lock_hash = sha256_file(lock_source)
    figure_hash = sha256_file(figure_source)
    if summary_hash != args.expected_summary_sha256 or lock_hash != args.expected_lock_sha256:
        raise RuntimeError("Objective 5 final protected hash changed")
    if summary_checksum.read_text(encoding="utf-8").split()[0] != summary_hash:
        raise RuntimeError("Objective 5 summary checksum mismatch")
    if figure_checksum.read_text(encoding="utf-8").split()[0] != figure_hash:
        raise RuntimeError("Objective 5 figure checksum mismatch")
    summary = json.loads(summary_source.read_text(encoding="utf-8"))
    lock = json.loads(lock_source.read_text(encoding="utf-8"))
    validate(summary, lock)
    if lock.get("summary_sha256") != summary_hash or lock.get("figure_sha256") != figure_hash:
        raise RuntimeError("Objective 5 final lock does not identify its artifacts")

    result_root = ROOT / args.result_path
    expected_names = {
        "README.md",
        "artifact_inventory_public.json",
        "locked_test_evaluation_lock_public.json",
        SUMMARY_NAME,
        f"{SUMMARY_NAME}.sha256",
        FIGURE_NAME,
        f"{FIGURE_NAME}.sha256",
    }
    result_root.mkdir(parents=True, exist_ok=True)
    for source in (summary_source, summary_checksum, figure_source, figure_checksum):
        target = result_root / source.name
        if target.is_file() and sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"Existing public artifact differs: {target}")
        shutil.copy2(source, target)
    public_lock = {
        key: lock[key]
        for key in (
            "artifact",
            "version",
            "manifest_sha256",
            "checkpoint_sha256",
            "selection_summary_sha256",
            "selection_lock_sha256",
            "protocol_sha256",
            "summary_sha256",
            "figure_sha256",
            "completed_datasets",
            "temperatures_reused_without_change",
            "thresholds_reused_without_change",
            "test_used_for_model_selection",
            "test_evaluation_count_per_dataset",
            "test_evaluated",
            "immutable",
        )
    }
    atomic_json(public_lock, result_root / "locked_test_evaluation_lock_public.json")
    (result_root / "README.md").write_text(
        "# Objective 5 final external-domain evaluation\n\n"
        "This directory reports the single locked-test evaluation of the frozen, "
        "validation-selected and validation-calibrated DenseNet-121 candidates on "
        "CheXpert and PadChest. Temperatures and decision thresholds were reused "
        "without change. Confidence intervals use patient-level bootstrap resampling.\n\n"
        "The six-positive Pneumothorax results require cautious interpretation. No "
        "patient/image identifiers, medical images, private manifests, checkpoints, "
        "or case-level predictions are included.\n",
        encoding="utf-8",
    )
    inventory = {
        "artifact": "Objective 5 final public artifact inventory",
        "summary_sha256": summary_hash,
        "final_lock_sha256": lock_hash,
        "figure_sha256": figure_hash,
        "private_manifests_included": False,
        "patient_identifiers_included": False,
        "image_identifiers_included": False,
        "medical_images_included": False,
        "case_level_predictions_included": False,
        "checkpoints_included": False,
        "files": {},
    }
    for path in sorted(result_root.iterdir()):
        if path.name != "artifact_inventory_public.json":
            inventory["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(inventory, result_root / "artifact_inventory_public.json")
    actual_names = {path.name for path in result_root.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(f"Unexpected Objective 5 public files: {sorted(actual_names ^ expected_names)}")
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.suffix.lower() in {".json", ".md", ".sha256"}
    )
    forbidden = ('"patient_id"', '"image_id"', '"image_path"', '"mask_path"')
    violations = [value for value in forbidden if value in serialized]
    if violations:
        raise RuntimeError(f"Objective 5 final privacy scan failed: {violations}")

    archive = Path("/kaggle/working/backups/objective5_external_locked_test_public_v1.0.0.zip")
    archive_hash = deterministic_zip(list(result_root.iterdir()), archive)
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    hf_api = HfApi(token=hf_token)
    if bool(hf_api.model_info(args.hf_repo, token=hf_token).private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    remote_files = set(hf_api.list_repo_files(args.hf_repo, repo_type="model", token=hf_token))
    operations = []
    verification_root = Path("/kaggle/working/objective5_final_hf_verify")
    for path in [*result_root.iterdir(), archive, archive_checksum]:
        remote = f"{args.hf_path.strip('/')}/{path.name}"
        if remote in remote_files:
            downloaded = Path(hf_hub_download(args.hf_repo, filename=remote, repo_type="model", token=hf_token, local_dir=verification_root, force_download=True))
            if sha256_file(downloaded) != sha256_file(path):
                raise RuntimeError(f"Existing HF artifact differs: {remote}")
        else:
            operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)))
    hf_commit = None
    if operations:
        hf_commit = hf_api.create_commit(
            repo_id=args.hf_repo,
            repo_type="model",
            token=hf_token,
            operations=operations,
            commit_message="results: publish Objective 5 final external evaluation",
        )

    run_git(["config", "user.name", "Ahmed Zuhair"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    result_prefix = str(args.result_path).replace("\\", "/").rstrip("/")
    status = run_git(["status", "--porcelain"])
    unexpected = []
    for line in status.splitlines():
        changed_path = line[3:].strip().split(" -> ")[-1].rstrip("/")
        inside = changed_path == result_prefix or changed_path.startswith(f"{result_prefix}/")
        parent = result_prefix.startswith(f"{changed_path}/")
        if not inside and not parent:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(f"Unexpected Git changes: {unexpected}")
    run_git(["add", "--", result_prefix])
    staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
    if staged and any(not path.startswith(f"{result_prefix}/") for path in staged):
        raise RuntimeError(f"Unexpected staged files: {staged}")
    if staged:
        run_git(["commit", "-m", "results: publish Objective 5 final external evaluation"])
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
        run_git(["push", "origin", "main"], environment)
    github_commit = run_git(["rev-parse", "HEAD"])

    response = github_request("GET", f"https://api.github.com/repos/{args.github_repo}/releases/tags/{args.release_tag}", github_token)
    if response.status_code == 404:
        response = github_request(
            "POST",
            f"https://api.github.com/repos/{args.github_repo}/releases",
            github_token,
            json={
                "tag_name": args.release_tag,
                "target_commitish": github_commit,
                "name": args.release_title,
                "body": "Final aggregate Objective 5 external-domain locked-test results.",
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
            uploaded = github_request("POST", upload_url, github_token, params={"name": asset.name}, headers={"Content-Type": "application/octet-stream"}, data=stream)
        uploaded.raise_for_status()
    print(
        json.dumps(
            {
                "summary_sha256": summary_hash,
                "final_lock_sha256": lock_hash,
                "figure_sha256": figure_hash,
                "public_archive_sha256": archive_hash,
                "hf_commit": getattr(hf_commit, "oid", None),
                "hf_path": f"https://huggingface.co/{args.hf_repo}/tree/main/{args.hf_path.strip('/')}",
                "github_commit": github_commit,
                "github_results": f"https://github.com/{args.github_repo}/tree/main/{result_prefix}",
                "github_release": release["html_url"],
                "privacy_scan_passed": True,
                "test_evaluation_count_per_dataset": 1,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 5 FINAL EXTERNAL-DOMAIN RESULTS PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
