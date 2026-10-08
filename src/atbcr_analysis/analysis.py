"""Analysis helpers for comparing final opinion distributions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from atbcr_analysis.metrics import MetricSummary, summarize_values


@dataclass(frozen=True, slots=True)
class PairedWassersteinResult:
    """Per-seed Wasserstein distances and summary statistics."""

    distances_by_seed: dict[int, float]
    summary: MetricSummary


@dataclass(frozen=True, slots=True)
class SeededFinalDistribution:
    """Final opinions and effective seeds for one run."""

    opinions: list[float]
    graph_seed: int | None = None
    opinion_seed: int | None = None
    dynamics_seed: int | None = None


SeededDistributionInput = Mapping[int, list[float] | SeededFinalDistribution]


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
    first: SeededDistributionInput,
    second: SeededDistributionInput,
) -> PairedWassersteinResult:
    """Compare two distributions seed by seed and summarize the distances."""

    first_seeds = set(first)
    second_seeds = set(second)
    if first_seeds != second_seeds:
        raise ValueError("paired Wasserstein comparison requires identical seed sets")
    distances: dict[int, float] = {}
    for seed in sorted(first_seeds):
        first_run = _as_seeded_distribution(first[seed])
        second_run = _as_seeded_distribution(second[seed])
        _validate_effective_seeds(seed, first_run, second_run)
        distances[seed] = wasserstein_distance_1d(first_run.opinions, second_run.opinions)
    return PairedWassersteinResult(
        distances_by_seed=distances,
        summary=summarize_values(list(distances.values())),
    )


def pairwise_paired_wasserstein(
    final_distributions: dict[str, dict[int, list[float] | SeededFinalDistribution]],
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
) -> dict[str, dict[int, SeededFinalDistribution]]:
    """Collect final opinions keyed by variant-like result names and run seed."""

    distributions: dict[str, dict[int, SeededFinalDistribution]] = {}
    for result in results:  # type: ignore[operator]
        key = (
            f"{result.scenario_name}/{result.variant_name}/"
            f"{result.normalizer_name}/{result.normalization_every}"
        )
        distributions[key] = {
            simulation.seed: SeededFinalDistribution(
                opinions=list(simulation.final_opinions),
                graph_seed=simulation.graph_seed,
                opinion_seed=simulation.opinion_seed,
                dynamics_seed=simulation.dynamics_seed,
            )
            for simulation in result.simulations
        }
    return distributions


def _as_seeded_distribution(
    value: list[float] | SeededFinalDistribution,
) -> SeededFinalDistribution:
    if isinstance(value, SeededFinalDistribution):
        return value
    return SeededFinalDistribution(opinions=value)


def _validate_effective_seeds(
    run_seed: int,
    first: SeededFinalDistribution,
    second: SeededFinalDistribution,
) -> None:
    for seed_name in ("graph_seed", "opinion_seed", "dynamics_seed"):
        first_seed = getattr(first, seed_name)
        second_seed = getattr(second, seed_name)
        if first_seed is not None and second_seed is not None and first_seed != second_seed:
            raise ValueError(
                f"paired Wasserstein run_seed={run_seed} has mismatched {seed_name}: "
                f"{first_seed} != {second_seed}",
            )
