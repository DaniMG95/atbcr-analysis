"""Plot helpers for persisted experiment outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def plot_metric_trajectories(
    trajectories_csv: Path,
    output_path: Path,
    *,
    metric: str = "max_abs_opinion",
) -> None:
    """Plot one metric over time for each scenario/variant/normalization series."""

    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with trajectories_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (
                f"{row['scenario']}/{row['variant']}/"
                f"{row['normalizer']}/{row['normalization_every']}"
            )
            series[label].append((int(row["step"]), float(row[metric])))

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, points in series.items():
        ordered = sorted(points)
        ax.plot([step for step, _ in ordered], [value for _, value in ordered], label=label)
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_final_distribution(
    snapshots_csv: Path,
    output_path: Path,
    *,
    bins: int = 50,
) -> None:
    """Plot final opinion histograms from the persisted snapshot table."""

    import matplotlib.pyplot as plt

    rows: list[dict[str, str]] = []
    with snapshots_csv.open(newline="", encoding="utf-8") as handle:
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

    opinions_by_label: dict[str, list[float]] = defaultdict(list)
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
        opinions_by_label[label].append(float(row["opinion"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, opinions in opinions_by_label.items():
        ax.hist(opinions, bins=bins, alpha=0.35, density=True, label=label)
    ax.set_xlabel("opinion")
    ax.set_ylabel("density")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
