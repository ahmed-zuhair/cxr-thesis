"""Locked-test guards for the Objective 3 v2.0 study.

Every script in this study calls :func:`assert_no_locked_test` on each input path
before opening it. The v2.0 protocol permits exactly one locked-test evaluation,
performed by a single dedicated script, and only after the advancement rule has
been met. Nothing else may read that cohort.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

FORBIDDEN_PATH_MARKERS = (
    "locked_test",
    "test_manifest",
    "test_labels",
    "official_test",
)


class LockedTestAccessError(RuntimeError):
    """Raised when a path appears to reference the locked test cohort."""


def assert_no_locked_test(path: str | Path) -> Path:
    """Return the path unchanged, or raise if it references locked test data.

    The check is a case-insensitive substring match against
    :data:`FORBIDDEN_PATH_MARKERS`. It is deliberately conservative: a false
    positive costs a rename, a false negative costs the study.
    """

    target = Path(path)
    haystack = str(target).replace("\\", "/").lower()
    for marker in FORBIDDEN_PATH_MARKERS:
        if marker in haystack:
            raise LockedTestAccessError(
                f"Refusing to open {target}: the path contains '{marker}'. "
                "The Objective 3 v2.0 locked test may only be opened by the "
                "dedicated final-evaluation script, and only once."
            )
    return target


def assert_no_locked_tests(paths: Iterable[str | Path]) -> list[Path]:
    """Apply :func:`assert_no_locked_test` to every path."""

    return [assert_no_locked_test(path) for path in paths]


def require_existing(paths: Iterable[str | Path]) -> list[Path]:
    """Guard every path, then fail loudly if any of them is missing.

    Scripts must never silently fall back to synthetic data when an input is
    absent, so this raises with the full list of what could not be found.
    """

    checked = assert_no_locked_tests(paths)
    missing = [str(path) for path in checked if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required inputs are missing:\n" + "\n".join(missing)
        )
    return checked
