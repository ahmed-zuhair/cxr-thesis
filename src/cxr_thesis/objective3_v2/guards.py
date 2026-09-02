"""Locked-test guards for the Objective 3 v2.0 study.

Every script in this study calls :func:`assert_no_locked_test` on each input path
before opening it. The v2.0 protocol permits exactly one locked-test evaluation,
performed by a single dedicated script, and only after the advancement rule has
been met. Nothing else may read that cohort.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class LockedTestAuthorisation:
    """A recorded, checkable reason for the single permitted test evaluation.

    Opening the locked test is the one irreversible step in the study. Rather
    than letting any script bypass :func:`assert_no_locked_test` quietly, the
    bypass requires this object, and every field it carries is written into the
    final lock so the decision can be audited afterwards.
    """

    protocol_sha256: str
    advancement_rule: str
    hypothesis_passed: str
    evidence_sha256: str
    evaluations_permitted: int = 1

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def open_locked_test(
    path: str | Path,
    authorisation: LockedTestAuthorisation,
) -> Path:
    """Return a locked-test path, but only with a recorded authorisation.

    This is the only sanctioned way past :func:`assert_no_locked_test`. It
    validates the authorisation rather than trusting it: an empty hypothesis, a
    missing protocol hash, or more than one permitted evaluation all raise.
    """

    if not isinstance(authorisation, LockedTestAuthorisation):
        raise TypeError("A LockedTestAuthorisation is required to open the test")
    if not authorisation.protocol_sha256 or len(authorisation.protocol_sha256) != 64:
        raise LockedTestAccessError("Authorisation carries no valid protocol hash")
    if not authorisation.hypothesis_passed:
        raise LockedTestAccessError(
            "Authorisation names no passing hypothesis; the advancement rule "
            "was not met and the locked test must stay closed"
        )
    if authorisation.evaluations_permitted != 1:
        raise LockedTestAccessError(
            "The protocol permits exactly one locked-test evaluation"
        )
    if not authorisation.evidence_sha256:
        raise LockedTestAccessError("Authorisation carries no evidence hash")
    return Path(path)


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
