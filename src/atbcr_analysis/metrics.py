"""Metrics for opinion dynamics experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Metrics captured at one simulation step."""

    step: int
    confidence_frequency: float
    inaction_frequency: float
    repulsion_frequency: float
    confidence_frequency_window: float
    inaction_frequency_window: float
    repulsion_frequency_window: float
    max_abs_opinion: float
    mean_abs_opinion: float
    std_opinion: float
    p50_abs_opinion: float
    p90_abs_opinion: float
    p95_abs_opinion: float
    p99_abs_opinion: float
    extremized_share: float
    cluster_count: int
    configured_cluster_tolerance: float
    effective_cluster_tolerance: float
    unbounded_extreme_cutoff: float


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Descriptive statistics and a normal-approximation 95% CI for the mean."""

    mean: float
    std: float
    median: float
    ci95_low: float
    ci95_high: float


@dataclass(slots=True)
class RuleCounts:
    """Cumulative interaction outcomes."""

    confidence: int = 0
    inaction: int = 0
    repulsion: int = 0

    @property
    def total(self) -> int:
        return self.confidence + self.inaction + self.repulsion

    def add(self, outcome: str) -> None:
        if outcome == "confidence":
            self.confidence += 1
        elif outcome == "inaction":
            self.inaction += 1
        elif outcome == "repulsion":
            self.repulsion += 1
        else:
            raise ValueError(f"Unsupported interaction outcome: {outcome}")

    def difference(self, previous: RuleCounts) -> RuleCounts:
        """Return counts accumulated since a previous snapshot."""

        return RuleCounts(
            confidence=self.confidence - previous.confidence,
            inaction=self.inaction - previous.inaction,
            repulsion=self.repulsion - previous.repulsion,
        )


def concern_share(opinions: list[float], threshold: float) -> float:
    """Return the proportion of agents with opinion above the concern threshold."""

    if not opinions:
        raise ValueError("opinions cannot be empty")
    return sum(opinion >= threshold for opinion in opinions) / len(opinions)


def mean(values: list[float]) -> float:
    """Return the arithmetic mean."""

    if not values:
        raise ValueError("values cannot be empty")
    return sum(values) / len(values)


def population_std(values: list[float]) -> float:
    """Return the population standard deviation."""

    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def sample_std(values: list[float]) -> float:
    """Return the sample standard deviation."""

    if not values:
        raise ValueError("values cannot be empty")
    if len(values) == 1:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def max_abs(values: list[float]) -> float:
    """Return max(abs(x))."""

    if not values:
        raise ValueError("values cannot be empty")
    return max(abs(value) for value in values)


def mean_abs(values: list[float]) -> float:
    """Return mean(abs(x))."""

    if not values:
        raise ValueError("values cannot be empty")
    return mean([abs(value) for value in values])


def percentile(values: list[float], percent: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""

    if not values:
        raise ValueError("values cannot be empty")
    if not 0 <= percent <= 100:
        raise ValueError("percent must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def abs_percentiles(values: list[float]) -> dict[str, float]:
    """Return standard percentiles of abs(values) used by the experiments."""

    absolute = [abs(value) for value in values]
    return {
        "p50_abs_opinion": percentile(absolute, 50),
        "p90_abs_opinion": percentile(absolute, 90),
        "p95_abs_opinion": percentile(absolute, 95),
        "p99_abs_opinion": percentile(absolute, 99),
    }


def summarize_values(values: list[float]) -> MetricSummary:
    """Summarize values using ``mean +/- 1.96 * sample_std / sqrt(n)`` for CI95."""

    if not values:
        raise ValueError("values cannot be empty")
    avg = mean(values)
    std = sample_std(values)
    if len(values) == 1:
        ci95_low = math.nan
        ci95_high = math.nan
    else:
        margin = 1.96 * std / (len(values) ** 0.5)
        ci95_low = avg - margin
        ci95_high = avg + margin
    return MetricSummary(
        mean=avg,
        std=std,
        median=percentile(values, 50),
        ci95_low=ci95_low,
        ci95_high=ci95_high,
    )


def extremized_share(
    opinions: list[float],
    unbounded_extreme_cutoff: float | None = None,
    domain: str = "bounded_01",
    *,
    threshold: float | None = None,
) -> float:
    """Return the share of agents beyond the domain-specific extreme cutoff.

    For ``bounded_01`` the ATBCR article defines extremes as ``[0, 0.1]`` and
    ``[0.9, 1]``. Under ``y = 2x - 1`` this maps exactly to ``abs(y) >= 0.8``.
    The unbounded domain has no natural endpoint, so its cutoff is supplied as
    ``unbounded_extreme_cutoff``. The keyword ``threshold`` is accepted as a
    temporary compatibility alias.
    """

    if not opinions:
        raise ValueError("opinions cannot be empty")
    if domain == "bounded_01":
        return sum(opinion <= 0.1 or opinion >= 0.9 for opinion in opinions) / len(opinions)
    if domain == "bounded_m11":
        return sum(abs(opinion) >= 0.8 for opinion in opinions) / len(opinions)
    if domain == "unbounded":
        if unbounded_extreme_cutoff is None:
            unbounded_extreme_cutoff = threshold
        if unbounded_extreme_cutoff is None:
            raise ValueError("unbounded_extreme_cutoff is required for unbounded domain")
        if unbounded_extreme_cutoff < 0:
            raise ValueError("unbounded_extreme_cutoff must be non-negative")
        return sum(abs(opinion) >= unbounded_extreme_cutoff for opinion in opinions) / len(opinions)
    raise ValueError(f"Unsupported domain: {domain}")


def cluster_count(opinions: list[float], tolerance: float) -> int:
    """Count clusters using a sorted anchor-based one-dimensional rule.

    The first sorted opinion starts the first cluster and becomes its anchor.
    Each following opinion joins the current cluster if its distance from that
    anchor is at most ``tolerance``. Otherwise it starts a new cluster and
    becomes the new anchor. This is intentionally not a consecutive-gap rule.
    """

    if not opinions:
        raise ValueError("opinions cannot be empty")
    if tolerance < 0:
        raise ValueError("cluster tolerance must be non-negative")

    sorted_opinions = sorted(opinions)
    clusters = 1
    current_cluster_anchor = sorted_opinions[0]
    for opinion in sorted_opinions[1:]:
        if abs(opinion - current_cluster_anchor) > tolerance:
            clusters += 1
            current_cluster_anchor = opinion
    return clusters


def effective_cluster_tolerance(configured_tolerance: float, domain: str) -> float:
    """Return the tolerance in the metric scale of the configured domain."""

    if configured_tolerance < 0:
        raise ValueError("cluster tolerance must be non-negative")
    if domain == "bounded_01":
        return configured_tolerance
    if domain in {"bounded_m11", "unbounded"}:
        return configured_tolerance * 2
    raise ValueError(f"Unsupported domain: {domain}")


def summarize_step(
    step: int,
    opinions: list[float],
    counts: RuleCounts,
    window_counts: RuleCounts,
    *,
    window_size: int,
    unbounded_extreme_cutoff: float,
    cluster_tolerance: float,
    domain: str,
) -> StepMetrics:
    """Build the standard metrics record for one snapshot."""

    total = max(1, counts.total)
    window_denominator = max(1, window_size)
    percentiles = abs_percentiles(opinions)
    effective_tolerance = effective_cluster_tolerance(cluster_tolerance, domain)
    return StepMetrics(
        step=step,
        confidence_frequency=counts.confidence / total,
        inaction_frequency=counts.inaction / total,
        repulsion_frequency=counts.repulsion / total,
        confidence_frequency_window=window_counts.confidence / window_denominator,
        inaction_frequency_window=window_counts.inaction / window_denominator,
        repulsion_frequency_window=window_counts.repulsion / window_denominator,
        max_abs_opinion=max_abs(opinions),
        mean_abs_opinion=mean_abs(opinions),
        std_opinion=population_std(opinions),
        **percentiles,
        extremized_share=extremized_share(opinions, unbounded_extreme_cutoff, domain),
        cluster_count=cluster_count(opinions, effective_tolerance),
        configured_cluster_tolerance=cluster_tolerance,
        effective_cluster_tolerance=effective_tolerance,
        unbounded_extreme_cutoff=unbounded_extreme_cutoff,
    )


def mape(predicted: list[float], observed: list[float]) -> float:
    """Return mean absolute percentage error in percent."""

    if len(predicted) != len(observed):
        raise ValueError("predicted and observed must have the same length")
    if any(value == 0 for value in observed):
        raise ValueError("observed values must be non-zero for MAPE")
    return 100 * mean([abs(p - o) / o for p, o in zip(predicted, observed, strict=True)])
