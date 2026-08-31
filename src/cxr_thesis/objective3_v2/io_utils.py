"""Hashing, atomic writes, and shard resume logic for Objective 3 v2.0.

Kaggle sessions expire and crash, and ``/kaggle/working/outputs`` does not
survive. Every long run therefore writes completed work as individually hashed
shards so a restart can skip what is already finished instead of recomputing it.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .guards import assert_no_locked_test

RESULTS_SCHEMA_KEYS = (
    "study",
    "part",
    "config",
    "results",
    "artifact_hashes",
    "timestamp",
    "locked_test_accessed",
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of an in-memory payload."""

    return hashlib.sha256(payload).hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    """Confirm a protected input still hashes to the value the protocol froze."""

    actual = sha256_file(path)
    if actual.lower() != expected.strip().lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}\n  expected {expected}\n  actual   {actual}"
        )
    return actual


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for the results schema."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON through a temporary file so a crash cannot truncate the result."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    text = json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def read_json(path: str | Path) -> Any:
    """Read a JSON file after checking it is not locked test data."""

    return json.loads(assert_no_locked_test(path).read_text(encoding="utf-8"))


def write_results(
    path: str | Path,
    *,
    study: str,
    part: str,
    config: dict[str, Any],
    results: dict[str, Any],
    artifact_hashes: dict[str, str] | None = None,
    seed: int | None = None,
    locked_test_accessed: bool = False,
) -> tuple[Path, str]:
    """Write one results JSON in the study schema plus its ``.sha256`` sidecar.

    Returns the results path and its own digest.
    """

    payload: dict[str, Any] = {
        "study": study,
        "part": part,
        "config": config,
        "results": results,
        "artifact_hashes": dict(artifact_hashes or {}),
        "timestamp": utc_timestamp(),
        "locked_test_accessed": bool(locked_test_accessed),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    missing = [key for key in RESULTS_SCHEMA_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Results payload is missing keys: {missing}")
    target = write_json_atomic(path, payload)
    digest = sha256_file(target)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return target, digest


def hash_directory(root: str | Path, pattern: str = "*") -> dict[str, str]:
    """Map each matching file under ``root`` to its digest, for an inventory."""

    base = Path(root)
    return {
        str(item.relative_to(base)).replace("\\", "/"): sha256_file(item)
        for item in sorted(base.rglob(pattern))
        if item.is_file() and not item.name.startswith(".")
    }


class ShardLedger:
    """Track completed shards of a long run so a restart can resume.

    A shard counts as complete only when its file exists *and* still matches the
    digest recorded when it was written, so a half-flushed file from a killed
    session is recomputed rather than trusted.
    """

    def __init__(self, index_path: str | Path, *, study: str, part: str) -> None:
        self.index_path = assert_no_locked_test(index_path)
        self.study = study
        self.part = part
        self._entries: dict[str, dict[str, Any]] = {}
        if self.index_path.exists():
            stored = read_json(self.index_path)
            if stored.get("study") == study and stored.get("part") == part:
                self._entries = {
                    str(record["shard"]): record
                    for record in stored.get("shards", [])
                }

    @property
    def completed(self) -> list[str]:
        """Names of shards recorded as finished, in insertion order."""

        return list(self._entries)

    def is_complete(self, shard: str, shard_path: str | Path) -> bool:
        """True when the shard is recorded and its file still matches its hash."""

        record = self._entries.get(str(shard))
        if record is None:
            return False
        target = Path(shard_path)
        if not target.is_file():
            return False
        try:
            return sha256_file(target) == record.get("sha256")
        except OSError:
            return False

    def mark_complete(
        self,
        shard: str,
        shard_path: str | Path,
        **extra: Any,
    ) -> dict[str, Any]:
        """Record a finished shard and flush the index to disk immediately."""

        target = Path(shard_path)
        if not target.is_file():
            raise FileNotFoundError(f"Cannot record a missing shard: {target}")
        record: dict[str, Any] = {
            "shard": str(shard),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
            "timestamp": utc_timestamp(),
        }
        record.update(extra)
        self._entries[str(shard)] = record
        self.flush()
        return record

    def flush(self) -> Path:
        """Persist the ledger atomically."""

        return write_json_atomic(
            self.index_path,
            {
                "study": self.study,
                "part": self.part,
                "timestamp": utc_timestamp(),
                "shards": list(self._entries.values()),
            },
        )
