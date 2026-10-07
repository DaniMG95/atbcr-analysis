"""Metrics for opinion dynamics experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Metrics captured at one simulation step."""

    step: int
    confidence_frequency: float
    inaction_frequency: float
    repulsion_frequency: float
    max_abs_opinion: float
    mean_abs_opinion: float
    std_opinion: float
    extremized_share: float
    cluster_count: int


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


def extremized_share(opinions: list[float], threshold: float, domain: str) -> float:
    """Return the share of agents close enough to the domain extremes."""

    if not opinions:
        raise ValueError("opinions cannot be empty")
    if threshold < 0:
        raise ValueError("extremized threshold must be non-negative")
    if domain == "bounded_01":
        lower = (1 - threshold) / 2
        upper = 1 - lower
        return sum(opinion <= lower or opinion >= upper for opinion in opinions) / len(opinions)
    if domain in {"bounded_m11", "unbounded"}:
        return sum(abs(opinion) >= threshold for opinion in opinions) / len(opinions)
    raise ValueError(f"Unsupported domain: {domain}")


def cluster_count(opinions: list[float], tolerance: float) -> int:
    """Count opinion clusters by grouping sorted opinions within a tolerance."""

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


def summarize_step(
    step: int,
    opinions: list[float],
    counts: RuleCounts,
    *,
    extremized_threshold: float,
    cluster_tolerance: float,
    domain: str,
) -> StepMetrics:
    """Build the standard metrics record for one snapshot."""

    total = max(1, counts.total)
    return StepMetrics(
        step=step,
        confidence_frequency=counts.confidence / total,
        inaction_frequency=counts.inaction / total,
        repulsion_frequency=counts.repulsion / total,
        max_abs_opinion=max_abs(opinions),
        mean_abs_opinion=mean_abs(opinions),
        std_opinion=population_std(opinions),
        extremized_share=extremized_share(opinions, extremized_threshold, domain),
        cluster_count=cluster_count(opinions, cluster_tolerance),
    )


def mape(predicted: list[float], observed: list[float]) -> float:
    """Return mean absolute percentage error in percent."""

    if len(predicted) != len(observed):
        raise ValueError("predicted and observed must have the same length")
    if any(value == 0 for value in observed):
        raise ValueError("observed values must be non-zero for MAPE")
    return 100 * mean([abs(p - o) / o for p, o in zip(predicted, observed, strict=True)])
