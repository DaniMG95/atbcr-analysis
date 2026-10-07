"""CLI for comparing persisted final opinion distributions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from atbcr_analysis.analysis import pairwise_wasserstein


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atbcr-compare",
        description="Compare final ATBCR opinion distributions from snapshots.csv.",
    )
    parser.add_argument("snapshots_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("wasserstein.csv"))
    args = parser.parse_args(argv)

    distributions = _read_final_distributions(args.snapshots_csv)
    distances = pairwise_wasserstein(distributions)
    _write_distances(args.output, distances)
    return 0


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
        label = f"{row['scenario']}/{row['variant']}/{row['normalizer']}/{row['normalization_every']}"
        distributions[label].append(float(row["opinion"]))
    return dict(distributions)


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


if __name__ == "__main__":
    raise SystemExit(main())
