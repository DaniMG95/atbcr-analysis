"""Analysis helpers for comparing final opinion distributions."""

from __future__ import annotations


def wasserstein_distance_1d(first: list[float], second: list[float]) -> float:
    """Return the empirical 1D Wasserstein distance for equally weighted samples."""

    if not first or not second:
        raise ValueError("distributions cannot be empty")
    if len(first) != len(second):
        raise ValueError("paired Wasserstein comparison requires equal sample sizes")
    ordered_first = sorted(first)
    ordered_second = sorted(second)
    return sum(abs(a - b) for a, b in zip(ordered_first, ordered_second, strict=True)) / len(first)


def pairwise_wasserstein(final_distributions: dict[str, list[float]]) -> dict[tuple[str, str], float]:
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


def final_distributions_by_variant(results: object) -> dict[str, list[float]]:
    """Collect final opinions keyed by variant-like result names."""

    distributions: dict[str, list[float]] = {}
    for result in results:  # type: ignore[operator]
        key = f"{result.scenario_name}/{result.variant_name}/{result.normalizer_name}/{result.normalization_every}"
        distributions[key] = [
            opinion
            for simulation in result.simulations
            for opinion in simulation.final_opinions
        ]
    return distributions
