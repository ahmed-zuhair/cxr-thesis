#!/usr/bin/env python3
"""Publish the sanitized Objective 3 final-evaluation protocol lock."""

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
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--validation-checksum", type=Path, required=True)
    parser.add_argument("--protocol-amendment", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--expected-final-protocol-sha256", required=True)
    parser.add_argument("--expected-final-cohort-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--expected-amendment-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--github-repo", required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_git(arguments: list[str], environment=None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=REPOSITORY_ROOT, env=environment,
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def deterministic_zip(files: list[Path], target: Path) -> str:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for source in sorted(files, key=lambda item: item.name):
            info = zipfile.ZipInfo(source.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, source.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Protocol archive integrity test failed")
    return sha256_file(target)


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not hf_token or not github_token:
        raise RuntimeError("HF_TOKEN and GITHUB_TOKEN must both be loaded")
    sources = [
        args.protocol_record, args.protocol_checksum, args.validation_summary,
        args.validation_checksum, args.protocol_amendment, args.frozen_config,
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Publication source files missing:\n" + "\n".join(missing))
    hashes = {
        "protocol": sha256_file(args.protocol_record),
        "validation": sha256_file(args.validation_summary),
        "amendment": sha256_file(args.protocol_amendment),
        "config": sha256_file(args.frozen_config),
    }
    expected = {
        "protocol": args.expected_final_protocol_sha256,
        "validation": args.expected_validation_sha256,
        "amendment": args.expected_amendment_sha256,
        "config": args.expected_config_sha256,
    }
    if hashes != expected:
        raise RuntimeError(f"Publication hashes do not match: {hashes}")
    if args.protocol_checksum.read_text(encoding="utf-8").split()[0] != hashes["protocol"]:
        raise RuntimeError("Final protocol checksum does not match")
    if args.validation_checksum.read_text(encoding="utf-8").split()[0] != hashes["validation"]:
        raise RuntimeError("Validation checksum does not match")
    protocol = json.loads(args.protocol_record.read_text(encoding="utf-8"))
    validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    checks = {
        "cohort": protocol.get("final_cohort_manifest_sha256")
        == args.expected_final_cohort_sha256,
        "cases": protocol.get("final_cohort_cases") == 5_000,
        "zero_overlap": protocol.get("patient_overlap_with_prior_evaluation_cohorts") == 0,
        "selection_labels": protocol.get("selection_used_labels") is False,
        "advance": protocol.get("advance_rule_verified") is True,
        "unevaluated": protocol.get("final_evaluated") is False,
        "one_evaluation": protocol.get("final_evaluation_count_allowed") == 1,
        "status": protocol.get("status")
        == "locked before final-cohort label evaluation",
        "privacy": protocol.get("private_manifest_included") is False,
        "validation_advance": validation.get("advance_to_single_final_evaluation") is True,
        "validation_test_blind": validation.get("test_evaluated") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Objective 3 final protocol checks failed: {failed}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source_commit, "HEAD"],
        cwd=REPOSITORY_ROOT, check=False,
    ).returncode:
        raise RuntimeError("Declared source commit is not in repository history")
    if run_git(["status", "--porcelain"]):
        raise RuntimeError("Repository must be clean before protocol publication")

    result_root = REPOSITORY_ROOT / args.result_path
    result_root.mkdir(parents=True, exist_ok=False)
    copies = {
        "objective3_final_protocol_lock_public.json": args.protocol_record,
        "objective3_final_protocol_lock_public.json.sha256": args.protocol_checksum,
        "objective3_enhancement_validation_summary_public.json": args.validation_summary,
        "objective3_enhancement_validation_summary_public.json.sha256": args.validation_checksum,
        "objective3_enhancement_protocol_amendment_public.json": args.protocol_amendment,
        "nih_quantum_gat_v1_1.yaml": args.frozen_config,
    }
    for name, source in copies.items():
        shutil.copy2(source, result_root / name)
    readme = f"""# Objective 3 independent final-evaluation protocol

- Final cohort images: 5,000
- Final cohort patients: {protocol['final_cohort_patients']}
- Prior evaluation patients excluded: {protocol['excluded_unique_patients']}
- Patient overlap: 0
- Selection seed: 4042
- Final cohort SHA-256: `{args.expected_final_cohort_sha256}`
- Validation result SHA-256: `{hashes['validation']}`
- Final protocol SHA-256: `{hashes['protocol']}`

The enhanced quantum head passed its frozen validation advance rule. Complete
previously unused official NIH test patients were selected without using labels,
predictions, or risk scores. This protocol is published before the single final
paired evaluation. No private manifest, identifiers, images, labels, checkpoint,
or case-level prediction is included.
"""
    (result_root / "README.md").write_text(readme, encoding="utf-8")
    inventory = {
        "artifact": "Objective 3 final protocol public inventory",
        "final_cohort_manifest_sha256": args.expected_final_cohort_sha256,
        "protocol_sha256": hashes["protocol"],
        "validation_summary_sha256": hashes["validation"],
        "private_manifest_included": False,
        "identifiers_included": False,
        "medical_images_included": False,
        "case_level_predictions_included": False,
        "final_results_included": False,
        "files": {},
    }
    for path in sorted(result_root.iterdir()):
        if path.is_file() and path.name != "artifact_inventory_public.json":
            inventory["files"][path.name] = {
                "bytes": path.stat().st_size, "sha256": sha256_file(path)
            }
    inventory_path = result_root / "artifact_inventory_public.json"
    atomic_json(inventory, inventory_path)
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in result_root.iterdir()
        if path.is_file()
    )
    forbidden = ('"patient_id"', '"image_id"', '"image_path"', '"mask_path"')
    violations = [item for item in forbidden if item in serialized]
    if violations:
        raise RuntimeError(f"Public protocol privacy scan failed: {violations}")

    backups = Path("/kaggle/working/backups")
    backups.mkdir(parents=True, exist_ok=True)
    archive = backups / "objective3_final_protocol_public_v1.0.0.zip"
    archive_hash = deterministic_zip(list(result_root.iterdir()), archive)
    archive_checksum = archive.with_suffix(".zip.sha256")
    archive_checksum.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=hf_token)
    if bool(api.model_info(args.hf_repo, token=hf_token).private):
        raise RuntimeError("Public checkpoint repository is unexpectedly private")
    public_files = [path for path in result_root.iterdir() if path.is_file()]
    public_files.extend([archive, archive_checksum])
    commit = api.create_commit(
        repo_id=args.hf_repo, repo_type="model", token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{args.hf_path.strip('/')}/{path.name}",
                path_or_fileobj=str(path),
            ) for path in public_files
        ],
        commit_message="protocol: publish Objective 3 independent final lock",
    )

    run_git(["config", "user.name", "Ahmed Zuhair"])
    run_git(["config", "user.email", "ahmed-zuhair@users.noreply.github.com"])
    relative = str(args.result_path).replace("\\", "/")
    run_git(["add", "--", relative])
    staged = run_git(["diff", "--cached", "--name-only"]).splitlines()
    if any(not name.startswith(relative.rstrip("/") + "/") for name in staged):
        raise RuntimeError(f"Unexpected staged files: {staged}")
    if staged:
        run_git(["commit", "-m", "protocol: lock Objective 3 final cohort"])
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
    print(json.dumps({
        "final_cohort_manifest_sha256": args.expected_final_cohort_sha256,
        "protocol_sha256": hashes["protocol"],
        "validation_summary_sha256": hashes["validation"],
        "public_archive_sha256": archive_hash,
        "hf_commit": getattr(commit, "oid", None),
        "hf_path": f"https://huggingface.co/{args.hf_repo}/tree/main/{args.hf_path.strip('/')}",
        "github_commit": github_commit,
        "github_results": f"https://github.com/{args.github_repo}/tree/main/{relative}",
        "final_evaluated": False,
        "private_manifest_published": False,
        "privacy_scan_passed": True,
    }, indent=2, sort_keys=True))
    print("OBJECTIVE 3 FINAL EVALUATION PROTOCOL PUBLISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
