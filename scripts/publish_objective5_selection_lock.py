#!/usr/bin/env python3
"""Publish the Objective 5 pre-test selection and calibration lock."""

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
PROTOCOL_SHA256 = "f36064954f16f0831739cf048d223bd39aacf833cc86c3dbbde92ff3c7085dfb"
CHECKPOINTS = {
    "chexpert": "edcd5792c57f04bdbef88043a2a11e422b506bdc2f26cd96f13121f6a8029c12",
    "padchest": "109db89a723c6e2f24442cb5866bfcf4084e85083936cda91bce3b8ae4365d9d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-directory", type=Path, required=True)
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
    temporary.replace(path)


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
            raise RuntimeError("Objective 5 selection-lock archive is corrupt")
    return sha256_file(target)


def validate(summary: dict[str, object], lock: dict[str, object]) -> None:
    if summary.get("artifact") != "Objective 5 selected-candidate and calibration lock":
        raise RuntimeError("Unexpected Objective 5 selection artifact")
    if summary.get("status") != "locked before either external-domain test evaluation":
        raise RuntimeError("Objective 5 selection is not in the pre-test state")
    if summary.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("Objective 5 adaptation protocol hash changed")
    for field in (
        "test_manifests_opened",
        "test_labels_accessed",
        "test_evaluated",
        "test_used_for_model_selection",
    ):
        if summary.get(field) is not False:
            raise RuntimeError(f"Unsafe Objective 5 state: {field}")
    datasets = summary.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(CHECKPOINTS):
        raise RuntimeError("Objective 5 selected datasets changed")
    for name, checkpoint_hash in CHECKPOINTS.items():
        result = datasets[name]
        if not isinstance(result, dict):
            raise TypeError(f"Malformed {name} selection result")
        thresholds = result.get("frozen_thresholds")
        checks = (
            result.get("selected_candidate") == "adapted",
            result.get("candidate_checkpoint_sha256") == checkpoint_hash,
            isinstance(result.get("temperature"), (int, float)),
            float(result.get("temperature", 0.0)) > 0.0,
            isinstance(thresholds, list),
            len(thresholds) == 6 if isinstance(thresholds, list) else False,
            all(0.0 < float(value) < 1.0 for value in thresholds)
            if isinstance(thresholds, list)
            else False,
        )
        if not all(checks):
            raise RuntimeError(f"Invalid frozen selection for {name}")
    if lock.get("immutable") is not True or lock.get("test_evaluated") is not False:
        raise RuntimeError(
            "Final Objective 5 selection lock is not immutable/test-blind"
        )
    if lock.get("external_test_evaluation_count") != 0:
        raise RuntimeError("Objective 5 external tests were already evaluated")
    if lock.get("selected_candidates") != CHECKPOINTS:
        raise RuntimeError("Final lock checkpoint selection changed")


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")

    summary_source = (
        args.selection_directory / "objective5_selection_calibration_public.json"
    )
    summary_checksum = summary_source.with_suffix(".json.sha256")
    lock_source = args.selection_directory / "FINAL_OBJECTIVE5_SELECTION_LOCK.json"
    lock_checksum = lock_source.with_suffix(".json.sha256")
    for path in (summary_source, summary_checksum, lock_source, lock_checksum):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary_hash = sha256_file(summary_source)
    lock_hash = sha256_file(lock_source)
    if summary_hash != args.expected_summary_sha256:
        raise RuntimeError("Objective 5 selection summary hash changed")
    if lock_hash != args.expected_lock_sha256:
        raise RuntimeError("Objective 5 final selection lock hash changed")
    if summary_checksum.read_text(encoding="utf-8").split()[0] != summary_hash:
        raise RuntimeError("Objective 5 summary checksum mismatch")
    if lock_checksum.read_text(encoding="utf-8").split()[0] != lock_hash:
        raise RuntimeError("Objective 5 lock checksum mismatch")
    summary = json.loads(summary_source.read_text(encoding="utf-8"))
    lock = json.loads(lock_source.read_text(encoding="utf-8"))
    validate(summary, lock)
    if lock.get("summary_sha256") != summary_hash:
        raise RuntimeError("Final lock does not identify the selection summary")

    result_root = ROOT / args.result_path
    expected_names = {
        summary_source.name,
        summary_checksum.name,
        lock_source.name,
        lock_checksum.name,
        "README.md",
        "artifact_inventory_public.json",
    }
    if result_root.exists():
        existing_names = {path.name for path in result_root.iterdir() if path.is_file()}
        if existing_names != expected_names:
            raise RuntimeError("Partial or unexpected Objective 5 publication exists")
    else:
        result_root.mkdir(parents=True)
        for source in (summary_source, summary_checksum, lock_source, lock_checksum):
            shutil.copy2(source, result_root / source.name)
        (result_root / "README.md").write_text(
            "# Objective 5 pre-test model-selection and calibration lock\n\n"
            "Both target-adapted DenseNet-121 candidates passed the locked "
            "validation advancement rule. One scalar temperature and six "
            "decision thresholds per dataset were fitted only on target "
            "validation data and frozen before locked-test access. No patient "
            "or image identifiers, medical images, private manifests, or "
            "case-level predictions are included.\n",
            encoding="utf-8",
        )
        inventory = {
            "artifact": "Objective 5 pre-test selection public inventory",
            "summary_sha256": summary_hash,
            "final_lock_sha256": lock_hash,
            "private_manifests_included": False,
            "patient_identifiers_included": False,
            "image_identifiers_included": False,
            "medical_images_included": False,
            "case_level_predictions_included": False,
            "files": {},
        }
        for path in sorted(result_root.iterdir()):
            inventory["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        atomic_json(inventory, result_root / "artifact_inventory_public.json")

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.suffix.lower() in {".json", ".md"}
    )
    forbidden = ('"patient_id"', '"image_id"', '"image_path"', '"mask_path"')
    violations = [value for value in forbidden if value in serialized]
    if violations:
        raise RuntimeError(f"Objective 5 privacy scan failed: {violations}")

    archive = Path(
        "/kaggle/working/backups/objective5_selection_calibration_lock_public_v1.0.0.zip"
    )
    archive_hash = deterministic_zip(list(result_root.iterdir()), archive)
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")

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
        commit_message="protocol: publish Objective 5 pre-test selection lock",
    )

    run_git(["config", "user.name", "Ahmed Zuhair"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    result_prefix = str(args.result_path).replace("\\", "/").rstrip("/")
    status = run_git(["status", "--porcelain"])
    unexpected = []
    for line in status.splitlines():
        changed_path = line[3:].strip().split(" -> ")[-1].rstrip("/")
        inside_result = changed_path == result_prefix or changed_path.startswith(
            f"{result_prefix}/"
        )
        summarized_parent = result_prefix.startswith(f"{changed_path}/")
        if not inside_result and not summarized_parent:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(f"Unexpected Git changes: {unexpected}")
    if status:
        run_git(["add", "--", result_prefix])
        staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
        if not staged or any(
            not path.startswith(f"{result_prefix}/") for path in staged
        ):
            raise RuntimeError(f"Unexpected staged files: {staged}")
        run_git(
            ["commit", "-m", "protocol: publish Objective 5 pre-test selection lock"]
        )
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

    release_url = f"https://api.github.com/repos/{args.github_repo}/releases/tags/{args.release_tag}"
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
                "body": "Objective 5 validation-only selection and calibration lock, published before external locked-test evaluation.",
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
                "summary_sha256": summary_hash,
                "final_lock_sha256": lock_hash,
                "public_archive_sha256": archive_hash,
                "hf_commit": getattr(hf_commit, "oid", None),
                "hf_path": f"https://huggingface.co/{args.hf_repo}/tree/main/{args.hf_path.strip('/')}",
                "github_commit": github_commit,
                "github_results": f"https://github.com/{args.github_repo}/tree/main/{result_prefix}",
                "github_release": release["html_url"],
                "test_evaluated": False,
                "private_manifests_published": False,
                "privacy_scan_passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("OBJECTIVE 5 PRE-TEST SELECTION LOCK PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
