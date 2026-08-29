#!/usr/bin/env python3
"""Privately back up and publicly publish the Objective 6 pre-training lock."""

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
PRIVATE_NAMES = {
    "train_report_cohort_private.csv",
    "train_report_cohort_private.csv.sha256",
    "val_report_cohort_private.csv",
    "val_report_cohort_private.csv.sha256",
    "test_report_cohort_private.csv",
    "test_report_cohort_private.csv.sha256",
}
CORE_PUBLIC_NAMES = {
    "objective6_report_generation_protocol_public.json",
    "objective6_report_generation_protocol_public.json.sha256",
    "objective6_report_cohort_summary_public.json",
    "objective6_report_cohort_summary_public.json.sha256",
    "FINAL_OBJECTIVE6_PRETRAINING_LOCK.json",
    "FINAL_OBJECTIVE6_PRETRAINING_LOCK.json.sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-directory", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--private-hf-repo", required=True)
    parser.add_argument("--private-hf-path", required=True)
    parser.add_argument("--public-hf-repo", required=True)
    parser.add_argument("--public-hf-path", required=True)
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


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


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
            raise RuntimeError(f"Archive is corrupt: {target}")
    return sha256(target)


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


def verify_checksum(path: Path) -> str:
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not checksum.is_file():
        raise FileNotFoundError(checksum)
    calculated = sha256(path)
    if checksum.read_text(encoding="utf-8").split()[0] != calculated:
        raise RuntimeError(f"Checksum mismatch: {path}")
    return calculated


def validate(
    protocol: dict[str, object], summary: dict[str, object], lock: dict[str, object]
) -> None:
    if protocol.get("status") != (
        "locked before cohort materialisation, vocabulary fitting, training, and evaluation"
    ):
        raise RuntimeError("Objective 6 protocol was not pre-training locked")
    if summary.get("candidate_studies") != 42066:
        raise RuntimeError("Objective 6 candidate-study count changed")
    if summary.get("candidate_patients") != 25342:
        raise RuntimeError("Objective 6 candidate-patient count changed")
    if summary.get("objective5_patients_excluded") != 40000:
        raise RuntimeError("Objective 5 patient exclusion changed")
    overlap = summary.get("patient_overlap")
    if not isinstance(overlap, dict) or any(int(value) for value in overlap.values()):
        raise RuntimeError("Objective 6 patient leakage detected")
    for field in ("model_training_performed", "model_inference_performed", "locked_test_evaluated"):
        if summary.get(field) is not False:
            raise RuntimeError(f"Unsafe pre-training state: {field}")
    manifests = summary.get("private_manifest_sha256")
    if not isinstance(manifests, dict) or set(manifests) != {
        "train_report_cohort_private.csv",
        "val_report_cohort_private.csv",
        "test_report_cohort_private.csv",
    }:
        raise RuntimeError("Objective 6 private manifest inventory changed")
    if lock.get("immutable") is not True:
        raise RuntimeError("Objective 6 final lock is not immutable")
    if lock.get("training_started") is not False:
        raise RuntimeError("Objective 6 training already started")
    if lock.get("locked_test_evaluation_count") != 0:
        raise RuntimeError("Objective 6 locked test was already evaluated")


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")

    public_source = args.lock_directory / "public"
    private_source = args.lock_directory / "private"
    if not public_source.is_dir() or not private_source.is_dir():
        raise FileNotFoundError("Objective 6 lock directory is incomplete")
    if {path.name for path in public_source.iterdir() if path.is_file()} != CORE_PUBLIC_NAMES:
        raise RuntimeError("Unexpected Objective 6 public lock files")
    if {path.name for path in private_source.iterdir() if path.is_file()} != PRIVATE_NAMES:
        raise RuntimeError("Unexpected Objective 6 private lock files")

    protocol_path = public_source / "objective6_report_generation_protocol_public.json"
    summary_path = public_source / "objective6_report_cohort_summary_public.json"
    lock_path = public_source / "FINAL_OBJECTIVE6_PRETRAINING_LOCK.json"
    protocol_hash = verify_checksum(protocol_path)
    summary_hash = verify_checksum(summary_path)
    lock_hash = verify_checksum(lock_path)
    if protocol_hash != args.expected_protocol_sha256:
        raise RuntimeError("Objective 6 protocol hash changed")
    if summary_hash != args.expected_summary_sha256:
        raise RuntimeError("Objective 6 cohort-summary hash changed")
    if lock_hash != args.expected_lock_sha256:
        raise RuntimeError("Objective 6 final-lock hash changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    validate(protocol, summary, lock)
    if summary.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Cohort summary does not identify the protocol")
    if lock.get("protocol_sha256") != protocol_hash or lock.get("cohort_summary_sha256") != summary_hash:
        raise RuntimeError("Final lock does not identify its inputs")

    for name, expected in summary["private_manifest_sha256"].items():
        path = private_source / name
        if verify_checksum(path) != expected:
            raise RuntimeError(f"Private manifest hash changed: {name}")

    if not args.audit_summary.is_file():
        raise FileNotFoundError(args.audit_summary)
    audit_hash = verify_checksum(args.audit_summary)
    if audit_hash != protocol.get("feasibility_audit_sha256"):
        raise RuntimeError("Objective 6 audit hash changed")

    backups = Path("/kaggle/working/backups")
    private_archive = backups / "objective6_report_cohorts_private_v1.0.0.zip"
    private_archive_hash = deterministic_zip(list(private_source.iterdir()), private_archive)
    private_archive_checksum = private_archive.with_suffix(".zip.sha256")
    private_archive_checksum.write_text(
        f"{private_archive_hash}  {private_archive.name}\n", encoding="utf-8"
    )

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=hf_token)
    if not bool(api.model_info(args.private_hf_repo, token=hf_token).private):
        raise RuntimeError("Private recovery repository must remain private")
    private_files = list(private_source.iterdir()) + [private_archive, private_archive_checksum]
    private_commit = api.create_commit(
        repo_id=args.private_hf_repo,
        repo_type="model",
        token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.private_hf_path.strip('/')}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in private_files
        ],
        commit_message="recovery: back up Objective 6 private report cohorts",
    )

    result_root = ROOT / args.result_path
    if result_root.exists():
        raise FileExistsError(f"Public result path already exists: {result_root}")
    result_root.mkdir(parents=True)
    for path in public_source.iterdir():
        shutil.copy2(path, result_root / path.name)
    shutil.copy2(args.audit_summary, result_root / args.audit_summary.name)
    shutil.copy2(
        args.audit_summary.with_suffix(args.audit_summary.suffix + ".sha256"),
        result_root / f"{args.audit_summary.name}.sha256",
    )
    (result_root / "README.md").write_text(
        "# Objective 6 clinical report-generation protocol\n\n"
        "This directory preregisters patient-disjoint Spanish radiology-report "
        "generation on PadChest before vocabulary fitting or model training. All "
        "40,000 PadChest patients used by Objective 5 were excluded. Private "
        "reports, identifiers, manifests, images, and case-level outputs are not "
        "included. The locked test may be evaluated exactly once after validation "
        "selection is complete.\n",
        encoding="utf-8",
    )
    inventory: dict[str, object] = {
        "artifact": "Objective 6 public pre-training artifact inventory",
        "protocol_sha256": protocol_hash,
        "cohort_summary_sha256": summary_hash,
        "final_lock_sha256": lock_hash,
        "audit_sha256": audit_hash,
        "private_manifests_included": False,
        "raw_reports_included": False,
        "patient_or_image_identifiers_included": False,
        "medical_images_included": False,
        "case_level_predictions_included": False,
        "files": {},
    }
    for path in sorted(result_root.iterdir()):
        inventory["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(inventory, result_root / "artifact_inventory_public.json")

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result_root.iterdir()
        if path.suffix.lower() in {".json", ".md"}
    )
    forbidden = ('"patient_id"', '"image_id"', '"image_path"', '"report"', '"labels"')
    violations = [field for field in forbidden if field in serialized]
    if violations:
        raise RuntimeError(f"Objective 6 public privacy scan failed: {violations}")

    public_archive = backups / "objective6_report_generation_protocol_public_v1.0.0.zip"
    public_archive_hash = deterministic_zip(list(result_root.iterdir()), public_archive)
    public_archive_checksum = public_archive.with_suffix(".zip.sha256")
    public_archive_checksum.write_text(
        f"{public_archive_hash}  {public_archive.name}\n", encoding="utf-8"
    )
    if bool(api.model_info(args.public_hf_repo, token=hf_token).private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    public_files = list(result_root.iterdir()) + [public_archive, public_archive_checksum]
    public_commit = api.create_commit(
        repo_id=args.public_hf_repo,
        repo_type="model",
        token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.public_hf_path.strip('/')}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in public_files
        ],
        commit_message="protocol: publish Objective 6 report-generation lock",
    )

    run_git(["config", "user.name", "Ahmed Zuhair Sabah"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    result_prefix = str(args.result_path).replace("\\", "/").rstrip("/")
    status = run_git(["status", "--porcelain"])
    unexpected = [
        line for line in status.splitlines()
        if not line[3:].strip().split(" -> ")[-1].rstrip("/").startswith(result_prefix)
        and not result_prefix.startswith(line[3:].strip().rstrip("/") + "/")
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected Git changes: {unexpected}")
    run_git(["add", "--", result_prefix])
    staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
    if not staged or any(not name.startswith(f"{result_prefix}/") for name in staged):
        raise RuntimeError(f"Unexpected staged files: {staged}")
    run_git(["commit", "-m", "protocol: publish Objective 6 report-generation lock"])
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
                "body": "Objective 6 report-generation protocol and patient-disjoint cohort lock, published before model training.",
                "draft": False,
                "prerelease": False,
            },
        )
    response.raise_for_status()
    release = response.json()
    existing_assets = {asset["name"] for asset in release.get("assets", [])}
    upload_url = release["upload_url"].split("{")[0]
    for asset in (public_archive, public_archive_checksum):
        if asset.name in existing_assets:
            continue
        with asset.open("rb") as stream:
            uploaded = github_request(
                "POST", upload_url, github_token,
                params={"name": asset.name},
                headers={"Content-Type": "application/octet-stream"},
                data=stream,
            )
        uploaded.raise_for_status()

    print(json.dumps({
        "protocol_sha256": protocol_hash,
        "cohort_summary_sha256": summary_hash,
        "final_lock_sha256": lock_hash,
        "private_archive_sha256": private_archive_hash,
        "public_archive_sha256": public_archive_hash,
        "private_hf_commit": getattr(private_commit, "oid", None),
        "private_hf_path": f"https://huggingface.co/{args.private_hf_repo}/tree/main/{args.private_hf_path.strip('/')}",
        "public_hf_commit": getattr(public_commit, "oid", None),
        "public_hf_path": f"https://huggingface.co/{args.public_hf_repo}/tree/main/{args.public_hf_path.strip('/')}",
        "github_commit": github_commit,
        "github_results": f"https://github.com/{args.github_repo}/tree/main/{result_prefix}",
        "github_release": release["html_url"],
        "privacy_scan_passed": True,
        "training_performed": False,
        "locked_test_evaluated": False,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 6 PRIVATE COHORT RECOVERY AND PUBLIC PROTOCOL PUBLICATION SUCCESSFUL")


if __name__ == "__main__":
    main()
