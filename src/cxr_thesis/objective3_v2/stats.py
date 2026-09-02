"""Statistical tests for the Objective 3 v2.0 study.

The v1.1 amendment selected on a count of seed wins across three seeds. That
criterion has almost no power: a sign test with three paired observations cannot
produce a p-value below 0.25 whatever the effect size. This module provides the
paired tests, the equivalence test, and the power calculations that replace it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy import stats

Z_TWO_SIDED_95 = 1.959963984540054


def _paired_array(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    """Return the finite paired differences, or raise if the input is unusable."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("Paired samples must have the same shape")
    if left.ndim != 1:
        raise ValueError("Paired samples must be one-dimensional")
    differences = left - right
    if not np.isfinite(differences).all():
        raise ValueError("Paired differences must be finite")
    if differences.size < 3:
        raise ValueError("At least three pairs are required")
    return differences


@dataclass(frozen=True)
class PairedResult:
    """Outcome of a paired difference test."""

    pairs: int
    mean_difference: float
    standard_deviation: float
    confidence_interval_95: tuple[float, float]
    statistic: float
    p_value: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_ttest(
    first: Sequence[float],
    second: Sequence[float],
) -> PairedResult:
    """Paired t-test on ``first - second`` with a 95% confidence interval."""

    differences = _paired_array(first, second)
    count = differences.size
    mean = float(differences.mean())
    deviation = float(differences.std(ddof=1))
    statistic, p_value = stats.ttest_rel(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
    )
    if deviation == 0.0:
        interval = (mean, mean)
    else:
        interval = stats.t.interval(
            0.95,
            count - 1,
            loc=mean,
            scale=deviation / np.sqrt(count),
        )
    return PairedResult(
        pairs=count,
        mean_difference=mean,
        standard_deviation=deviation,
        confidence_interval_95=(float(interval[0]), float(interval[1])),
        statistic=float(statistic),
        p_value=float(p_value),
    )


def paired_wilcoxon(
    first: Sequence[float],
    second: Sequence[float],
) -> PairedResult:
    """Wilcoxon signed-rank test on ``first - second``.

    Reported alongside the t-test because macro AUROC differences across seeds
    are not guaranteed to be normal at small seed counts.
    """

    differences = _paired_array(first, second)
    count = differences.size
    mean = float(differences.mean())
    deviation = float(differences.std(ddof=1))
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = stats.wilcoxon(differences)
    if deviation == 0.0:
        interval = (mean, mean)
    else:
        interval = stats.t.interval(
            0.95,
            count - 1,
            loc=mean,
            scale=deviation / np.sqrt(count),
        )
    return PairedResult(
        pairs=count,
        mean_difference=mean,
        standard_deviation=deviation,
        confidence_interval_95=(float(interval[0]), float(interval[1])),
        statistic=float(statistic),
        p_value=float(p_value),
    )


@dataclass(frozen=True)
class EquivalenceResult:
    """Outcome of a two one-sided tests (TOST) equivalence procedure."""

    pairs: int
    margin: float
    alpha: float
    mean_difference: float
    standard_deviation: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def p_value_text(self) -> str:
        """Format the p-value without ever rounding a small one to zero.

        ``p = 0.0000`` claims infinite evidence and is never true. Below the
        printing resolution the honest statement is an inequality.
        """

        if self.p_tost < 1e-4:
            return "p < 0.0001"
        return f"p = {self.p_tost:.4f}"

    def sentence(self, label_a: str = "quantum", label_b: str = "classical") -> str:
        """Return the claim in the form it should appear in the thesis."""

        if self.equivalent:
            return (
                f"{label_a.capitalize()} and {label_b} were statistically "
                f"equivalent within ±{self.margin:g} macro AUROC "
                f"(TOST, {self.p_value_text}, n = {self.pairs} seeds)."
            )
        return (
            f"Equivalence between {label_a} and {label_b} within "
            f"±{self.margin:g} macro AUROC could NOT be established "
            f"(TOST, {self.p_value_text}, n = {self.pairs} seeds); the result "
            "is inconclusive rather than a demonstration of equivalence."
        )


def tost_equivalence(
    first: Sequence[float],
    second: Sequence[float],
    margin: float,
    alpha: float = 0.05,
) -> EquivalenceResult:
    """Two one-sided tests for equivalence of paired samples.

    Failing to reject a null of "no difference" is not evidence of equivalence.
    TOST reverses the burden: the null is that the true difference lies outside
    ``+/- margin``, so rejecting it supports a positive claim of equivalence.
    """

    if margin <= 0:
        raise ValueError("The equivalence margin must be positive")
    if not 0 < alpha < 0.5:
        raise ValueError("Alpha must lie in (0, 0.5)")
    differences = _paired_array(first, second)
    count = differences.size
    mean = float(differences.mean())
    deviation = float(differences.std(ddof=1))
    degrees = count - 1
    if deviation == 0.0:
        inside = abs(mean) < margin
        return EquivalenceResult(
            pairs=count,
            margin=float(margin),
            alpha=float(alpha),
            mean_difference=mean,
            standard_deviation=0.0,
            p_lower=0.0 if inside else 1.0,
            p_upper=0.0 if inside else 1.0,
            p_tost=0.0 if inside else 1.0,
            equivalent=bool(inside),
        )
    standard_error = deviation / np.sqrt(count)
    p_lower = float(stats.t.sf((mean + margin) / standard_error, degrees))
    p_upper = float(stats.t.cdf((mean - margin) / standard_error, degrees))
    p_tost = max(p_lower, p_upper)
    return EquivalenceResult(
        pairs=count,
        margin=float(margin),
        alpha=float(alpha),
        mean_difference=mean,
        standard_deviation=deviation,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=float(p_tost),
        equivalent=bool(p_tost < alpha),
    )


def paired_power(
    effect: float,
    standard_deviation: float,
    pairs: int,
    alpha: float = 0.05,
) -> float:
    """Exact power of a two-sided paired t-test using the noncentral t."""

    if standard_deviation <= 0 or pairs < 2:
        raise ValueError("A positive standard deviation and n >= 2 are required")
    degrees = pairs - 1
    critical = stats.t.ppf(1 - alpha / 2, degrees)
    noncentrality = abs(effect) * np.sqrt(pairs) / standard_deviation
    upper = stats.nct.sf(critical, degrees, noncentrality)
    lower = stats.nct.cdf(-critical, degrees, noncentrality)
    return float(upper + lower)


def min_detectable_effect(
    standard_deviation: float,
    pairs: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest paired difference detectable at the requested power.

    Solved exactly against the noncentral t distribution rather than the usual
    normal approximation, because the seed counts here are small enough that the
    two disagree noticeably.
    """

    if not 0 < power < 1:
        raise ValueError("Power must lie in (0, 1)")
    if standard_deviation <= 0:
        raise ValueError("The standard deviation must be positive")
    if pairs < 2:
        raise ValueError("At least two pairs are required")
    low, high = 0.0, 10.0 * standard_deviation
    for _ in range(200):
        middle = (low + high) / 2.0
        if paired_power(middle, standard_deviation, pairs, alpha) < power:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def required_pairs(
    effect: float,
    standard_deviation: float,
    alpha: float = 0.05,
    power: float = 0.8,
    maximum: int = 100_000,
) -> int:
    """Smallest number of paired seeds that detects ``effect`` at ``power``."""

    if effect <= 0:
        raise ValueError("The target effect must be positive")
    for pairs in range(2, maximum + 1):
        if paired_power(effect, standard_deviation, pairs, alpha) >= power:
            return pairs
    raise ValueError(
        f"More than {maximum} pairs would be required; the effect is too small "
        "relative to the observed variability."
    )


def mde_curve(
    standard_deviation: float,
    pair_counts: Sequence[int],
    alpha: float = 0.05,
    power: float = 0.8,
) -> list[dict[str, float]]:
    """Minimum detectable effect at each seed count, ready to plot or serialise."""

    return [
        {
            "pairs": int(pairs),
            "min_detectable_effect": min_detectable_effect(
                standard_deviation, int(pairs), alpha, power
            ),
        }
        for pairs in pair_counts
    ]


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate with a percentile bootstrap interval."""

    point: float
    confidence_interval_95: tuple[float, float]
    resamples: int
    p_value_report: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    data: Sequence[float] | np.ndarray,
    resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap interval for ``statistic`` over ``data``.

    ``p_value_report`` carries the smallest p-value this resample count can
    resolve. Reporting a bootstrap p-value as 0.0 overstates the evidence; the
    honest statement is ``p < 1/B``.
    """

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Bootstrap data must be a one-dimensional sample")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap data must be finite")
    if resamples < 100:
        raise ValueError("Use at least 100 resamples")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(resamples, values.size))
    estimates = np.array([float(statistic(values[row])) for row in indices])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return BootstrapResult(
        point=float(statistic(values)),
        confidence_interval_95=(float(low), float(high)),
        resamples=int(resamples),
        p_value_report=f"p < {1.0 / resamples:g}",
    )


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, preserving the input order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("A one-dimensional, non-empty p-value list is required")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("P-values must be finite and lie in [0, 1]")
    count = values.size
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for rank, index in enumerate(order[::-1]):
        position = count - rank
        running = min(running, values[index] * count / position)
        adjusted[index] = running
    return [float(value) for value in adjusted]
