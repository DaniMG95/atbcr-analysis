"""CLI for comparing persisted final opinion distributions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from atbcr_analysis.analysis import (
    PairedWassersteinResult,
    SeededFinalDistribution,
    paired_wasserstein_by_seed,
    wasserstein_distance_1d,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atbcr-compare",
        description="Compare final ATBCR opinion distributions from snapshots.csv.",
    )
    parser.add_argument("snapshots_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("wasserstein.csv"))
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Compare distributions after pooling all final opinions across seeds.",
    )
    parser.add_argument(
        "--cross-scenario",
        action="store_true",
        help="Allow comparisons between different scenarios.",
    )
    parser.add_argument(
        "--allow-cross-domain",
        action="store_true",
        help="Allow comparisons between variants that use different opinion domains.",
    )
    args = parser.parse_args(argv)

    if args.aggregate:
        distributions = _read_final_distributions(args.snapshots_csv)
        distances = _pairwise_compatible_wasserstein(
            distributions,
            cross_scenario=args.cross_scenario,
            allow_cross_domain=args.allow_cross_domain,
        )
        _write_distances(args.output, distances)
    else:
        distributions_by_seed = _read_final_distributions_by_seed(args.snapshots_csv)
        distances_by_seed = _pairwise_compatible_paired_wasserstein(
            distributions_by_seed,
            cross_scenario=args.cross_scenario,
            allow_cross_domain=args.allow_cross_domain,
        )
        _write_paired_distances(args.output, distances_by_seed)
        _write_paired_distances_by_seed(
            _by_seed_output_path(args.output),
            distances_by_seed,
        )
    return 0


@dataclass(frozen=True, slots=True)
class SeriesKey:
    """Parsed identity of one persisted result series."""

    label: str
    scenario: str
    variant: str
    normalizer: str
    normalization_every: str

    @property
    def domain(self) -> str:
        if self.variant == "baseline_01":
            return "bounded_01"
        if self.variant == "bounded_m11":
            return "bounded_m11"
        if self.variant == "unbounded":
            return "unbounded"
        return self.variant


def _read_final_distributions(path: Path) -> dict[str, list[float]]:
    rows: list[dict[str, str]]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("snapshots file is empty")

    max_step_by_run: dict[tuple[str, str, str, str, str], int] = {}
    for row in rows:
        key = (
            row["scenario"],
            row["variant"],
            row["normalizer"],
            row["normalization_every"],
            row["run_seed"],
        )
        max_step_by_run[key] = max(max_step_by_run.get(key, 0), int(row["step"]))

    distributions: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario"],
            row["variant"],
            row["normalizer"],
            row["normalization_every"],
            row["run_seed"],
        )
        if int(row["step"]) != max_step_by_run[key]:
            continue
        label = (
            f"{row['scenario']}/{row['variant']}/"
            f"{row['normalizer']}/{row['normalization_every']}"
        )
        distributions[label].append(float(row["opinion"]))
    return dict(distributions)


def _read_final_distributions_by_seed(path: Path) -> dict[str, dict[int, SeededFinalDistribution]]:
    rows: list[dict[str, str]]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("snapshots file is empty")

    max_step_by_run: dict[tuple[str, str, str, str, str], int] = {}
    for row in rows:
        key = (
            row["scenario"],
            row["variant"],
            row["normalizer"],
            row["normalization_every"],
            row["run_seed"],
        )
        max_step_by_run[key] = max(max_step_by_run.get(key, 0), int(row["step"]))

    opinions: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    effective_seeds: dict[tuple[str, int], tuple[int | None, int | None, int | None]] = {}
    for row in rows:
        key = (
            row["scenario"],
            row["variant"],
            row["normalizer"],
            row["normalization_every"],
            row["run_seed"],
        )
        if int(row["step"]) != max_step_by_run[key]:
            continue
        label = (
            f"{row['scenario']}/{row['variant']}/"
            f"{row['normalizer']}/{row['normalization_every']}"
        )
        run_seed = int(row["run_seed"])
        opinions[label][run_seed].append(float(row["opinion"]))
        effective_seeds[(label, run_seed)] = (
            _optional_int(row.get("graph_seed")),
            _optional_int(row.get("opinion_seed")),
            _optional_int(row.get("dynamics_seed")),
        )
    return {
        label: {
            run_seed: SeededFinalDistribution(
                opinions=values,
                graph_seed=effective_seeds[(label, run_seed)][0],
                opinion_seed=effective_seeds[(label, run_seed)][1],
                dynamics_seed=effective_seeds[(label, run_seed)][2],
            )
            for run_seed, values in seed_distributions.items()
        }
        for label, seed_distributions in opinions.items()
    }


def _write_distances(path: Path, distances: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["first", "second", "wasserstein_distance"])
        writer.writeheader()
        for (first, second), distance in distances.items():
            writer.writerow(
                {
                    "first": first,
                    "second": second,
                    "wasserstein_distance": distance,
                },
            )


def _pairwise_compatible_wasserstein(
    distributions: dict[str, list[float]],
    *,
    cross_scenario: bool,
    allow_cross_domain: bool,
) -> dict[tuple[str, str], float]:
    return {
        (first, second): wasserstein_distance_1d(
            distributions[first],
            distributions[second],
        )
        for first, second in _compatible_pairs(
            distributions,
            cross_scenario=cross_scenario,
            allow_cross_domain=allow_cross_domain,
        )
    }


def _pairwise_compatible_paired_wasserstein(
    distributions: dict[str, dict[int, SeededFinalDistribution]],
    *,
    cross_scenario: bool,
    allow_cross_domain: bool,
) -> dict[tuple[str, str], PairedWassersteinResult]:
    return {
        (first, second): paired_wasserstein_by_seed(
            distributions[first],
            distributions[second],
        )
        for first, second in _compatible_pairs(
            distributions,
            cross_scenario=cross_scenario,
            allow_cross_domain=allow_cross_domain,
        )
    }


def _compatible_pairs(
    distributions: dict[str, object],
    *,
    cross_scenario: bool,
    allow_cross_domain: bool,
) -> list[tuple[str, str]]:
    keys = {label: _parse_series_label(label) for label in distributions}
    labels = sorted(distributions)
    pairs: list[tuple[str, str]] = []
    for index, first_label in enumerate(labels):
        for second_label in labels[index + 1:]:
            if _series_are_compatible(
                keys[first_label],
                keys[second_label],
                cross_scenario=cross_scenario,
                allow_cross_domain=allow_cross_domain,
            ):
                pairs.append((first_label, second_label))
    return pairs


def _series_are_compatible(
    first: SeriesKey,
    second: SeriesKey,
    *,
    cross_scenario: bool,
    allow_cross_domain: bool,
) -> bool:
    if not cross_scenario and first.scenario != second.scenario:
        return False
    if not allow_cross_domain and first.domain != second.domain:
        return False
    return (
        first.variant,
        first.normalizer,
        first.normalization_every,
    ) != (
        second.variant,
        second.normalizer,
        second.normalization_every,
    )


def _parse_series_label(label: str) -> SeriesKey:
    parts = label.split("/")
    if len(parts) != 4:
        raise ValueError(f"Unsupported series label format: {label}")
    scenario, variant, normalizer, normalization_every = parts
    return SeriesKey(
        label=label,
        scenario=scenario,
        variant=variant,
        normalizer=normalizer,
        normalization_every=normalization_every,
    )


def _write_paired_distances(
    path: Path,
    distances: dict[tuple[str, str], PairedWassersteinResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "first",
                "second",
                "mean",
                "std",
                "median",
                "ci95_low",
                "ci95_high",
            ],
        )
        writer.writeheader()
        for (first, second), result in distances.items():
            summary = result.summary
            writer.writerow(
                {
                    "first": first,
                    "second": second,
                    "mean": summary.mean,
                    "std": summary.std,
                    "median": summary.median,
                    "ci95_low": summary.ci95_low,
                    "ci95_high": summary.ci95_high,
                },
            )


def _write_paired_distances_by_seed(
    path: Path,
    distances: dict[tuple[str, str], PairedWassersteinResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["first", "second", "run_seed", "wasserstein_distance"],
        )
        writer.writeheader()
        for (first, second), result in distances.items():
            for run_seed, distance in result.distances_by_seed.items():
                writer.writerow(
                    {
                        "first": first,
                        "second": second,
                        "run_seed": run_seed,
                        "wasserstein_distance": distance,
                    },
                )


def _by_seed_output_path(summary_path: Path) -> Path:
    return summary_path.with_name(f"{summary_path.stem}_by_seed{summary_path.suffix}")


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
