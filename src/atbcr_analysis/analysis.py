"""Analysis helpers for comparing final opinion distributions."""

from __future__ import annotations

from dataclasses import dataclass

from atbcr_analysis.metrics import MetricSummary, summarize_values


@dataclass(frozen=True, slots=True)
class PairedWassersteinResult:
    """Per-seed Wasserstein distances and summary statistics."""

    distances_by_seed: dict[int, float]
    summary: MetricSummary


def wasserstein_distance_1d(first: list[float], second: list[float]) -> float:
    """Return the empirical 1D Wasserstein distance for equally weighted samples."""

    if not first or not second:
        raise ValueError("distributions cannot be empty")
    if len(first) != len(second):
        raise ValueError("paired Wasserstein comparison requires equal sample sizes")
    ordered_first = sorted(first)
    ordered_second = sorted(second)
    total_distance = sum(
        abs(a - b)
        for a, b in zip(ordered_first, ordered_second, strict=True)
    )
    return total_distance / len(first)


def pairwise_wasserstein(
    final_distributions: dict[str, list[float]],
) -> dict[tuple[str, str], float]:
    """Compare every pair of named final distributions."""

    names = sorted(final_distributions)
    distances: dict[tuple[str, str], float] = {}
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            distances[(first_name, second_name)] = wasserstein_distance_1d(
                final_distributions[first_name],
                final_distributions[second_name],
            )
    return distances


def paired_wasserstein_by_seed(
    first: dict[int, list[float]],
    second: dict[int, list[float]],
) -> PairedWassersteinResult:
    """Compare two distributions seed by seed and summarize the distances."""

    first_seeds = set(first)
    second_seeds = set(second)
    if first_seeds != second_seeds:
        raise ValueError("paired Wasserstein comparison requires identical seed sets")
    distances = {
        seed: wasserstein_distance_1d(first[seed], second[seed])
        for seed in sorted(first_seeds)
    }
    return PairedWassersteinResult(
        distances_by_seed=distances,
        summary=summarize_values(list(distances.values())),
    )


def pairwise_paired_wasserstein(
    final_distributions: dict[str, dict[int, list[float]]],
) -> dict[tuple[str, str], PairedWassersteinResult]:
    """Compare every pair of named final distributions with seed pairing."""

    names = sorted(final_distributions)
    distances: dict[tuple[str, str], PairedWassersteinResult] = {}
    for index, first_name in enumerate(names):
        for second_name in names[index + 1:]:
            distances[(first_name, second_name)] = paired_wasserstein_by_seed(
                final_distributions[first_name],
                final_distributions[second_name],
            )
    return distances


def final_distributions_by_variant(results: object) -> dict[str, list[float]]:
    """Collect final opinions keyed by variant-like result names."""

    distributions: dict[str, list[float]] = {}
    for result in results:  # type: ignore[operator]
        key = (
            f"{result.scenario_name}/{result.variant_name}/"
            f"{result.normalizer_name}/{result.normalization_every}"
        )
        distributions[key] = [
            opinion
            for simulation in result.simulations
            for opinion in simulation.final_opinions
        ]
    return distributions


def final_distributions_by_variant_and_seed(
    results: object,
) -> dict[str, dict[int, list[float]]]:
    """Collect final opinions keyed by variant-like result names and run seed."""

    distributions: dict[str, dict[int, list[float]]] = {}
    for result in results:  # type: ignore[operator]
        key = (
            f"{result.scenario_name}/{result.variant_name}/"
            f"{result.normalizer_name}/{result.normalization_every}"
        )
        distributions[key] = {
            simulation.seed: list(simulation.final_opinions)
            for simulation in result.simulations
        }
    return distributions
