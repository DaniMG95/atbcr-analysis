"""Plot helpers for persisted experiment outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple


class SnapshotSeries(NamedTuple):
    scenario: str
    variant: str
    normalizer: str
    normalization_every: str
    run_seed: str


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
        label = (
            f"{row['scenario']}/{row['variant']}/"
            f"{row['normalizer']}/{row['normalization_every']}"
        )
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


def plot_opinion_trajectories(
    snapshots_csv: Path,
    output_path: Path,
    *,
    scenario: str | None = None,
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
    run_seed: str | None = None,
    max_agents: int | None = None,
) -> None:
    """Plot individual opinion trajectories for one persisted simulation run."""

    import matplotlib.pyplot as plt

    rows = _read_snapshot_rows(snapshots_csv)
    selected = _select_snapshot_series(
        rows,
        scenario=scenario,
        variant=variant,
        normalizer=normalizer,
        normalization_every=normalization_every,
        run_seed=run_seed,
    )
    trajectories = _build_agent_trajectories(rows, selected, max_agents=max_agents)
    if not trajectories:
        raise ValueError("No snapshot opinions matched the selected series")

    fig, ax = plt.subplots(figsize=(10, 6))
    for points in trajectories.values():
        ordered = sorted(points)
        ax.scatter(
            [step for step, _ in ordered],
            [opinion for _, opinion in ordered],
            color="blue",
            alpha=0.08,
            s=4,
        )

    initial = [points[0] for points in trajectories.values() if points]
    final = [points[-1] for points in trajectories.values() if points]
    ax.scatter(
        [step for step, _ in initial],
        [opinion for _, opinion in initial],
        color="green",
        s=10,
        label="initial opinions",
        zorder=3,
    )
    ax.scatter(
        [step for step, _ in final],
        [opinion for _, opinion in final],
        color="red",
        s=10,
        label="final opinions",
        zorder=3,
    )

    title = _build_opinion_plot_title(snapshots_csv, selected)
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("opinion")
    ax.grid(alpha=0.35)
    ax.legend(fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _read_snapshot_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("snapshots file is empty")
    return rows


def _build_opinion_plot_title(snapshots_csv: Path, selected: SnapshotSeries) -> str:
    metadata = _read_run_metadata(snapshots_csv.with_name("config.json"))
    scenario = _find_by_name(metadata.get("scenarios", []), selected.scenario)
    variant = _find_by_name(metadata.get("variants", []), selected.variant)

    model = scenario.get("model", {})
    parameters = _format_model_parameters(model, variant)
    opinion_range = _format_opinion_range(variant)
    normalizer = selected.normalizer
    if selected.normalization_every:
        normalizer = f"{normalizer} every {selected.normalization_every}"

    if parameters and opinion_range:
        return f"{parameters}; rango {opinion_range}; {normalizer}; seed {selected.run_seed}"
    if parameters:
        return f"{parameters}; {selected.variant}; {normalizer}; seed {selected.run_seed}"
    return (
        f"{selected.scenario}/{selected.variant}/"
        f"{selected.normalizer}/{selected.normalization_every} seed={selected.run_seed}"
    )


def _read_run_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return payload


def _find_by_name(items: Any, name: str) -> dict[str, Any]:
    if not isinstance(items, list):
        return {}
    for item in items:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return {}


def _format_model_parameters(model: dict[str, Any], variant: dict[str, Any]) -> str:
    if not model:
        return ""
    name = model.get("name", "model")
    if name == "atbcr":
        threshold_scale = float(variant.get("threshold_scale", 1.0) or 1.0)
        epsilon = _scale_optional_number(model.get("epsilon"), threshold_scale)
        theta = _scale_optional_number(model.get("theta"), threshold_scale)
        return (
            f"epsilon={epsilon}, "
            f"theta={theta}, "
            f"mu={model.get('mu')}"
        )
    return str(name)


def _scale_optional_number(value: Any, factor: float) -> Any:
    if isinstance(value, int | float):
        return value * factor
    return value


def _format_opinion_range(variant: dict[str, Any]) -> str:
    if not variant:
        return ""
    initializer = variant.get("initializer", {})
    if isinstance(initializer, dict) and {"low", "high"} <= set(initializer):
        return f"[{initializer['low']}, {initializer['high']}]"
    domain_ranges = {
        "bounded_01": "[0, 1]",
        "bounded_m11": "[-1, 1]",
        "unbounded": "unbounded",
    }
    return domain_ranges.get(str(variant.get("domain", "")), "")


def _select_snapshot_series(
    rows: list[dict[str, str]],
    *,
    scenario: str | None,
    variant: str | None,
    normalizer: str | None,
    normalization_every: str | None,
    run_seed: str | None,
) -> SnapshotSeries:
    candidates = sorted(
        {
            SnapshotSeries(
                row["scenario"],
                row["variant"],
                row["normalizer"],
                row["normalization_every"],
                row["run_seed"],
            )
            for row in rows
        },
    )
    filtered = [
        candidate
        for candidate in candidates
        if (scenario is None or candidate.scenario == scenario)
        and (variant is None or candidate.variant == variant)
        and (normalizer is None or candidate.normalizer == normalizer)
        and (
            normalization_every is None
            or candidate.normalization_every == normalization_every
        )
        and (run_seed is None or candidate.run_seed == run_seed)
    ]
    if not filtered:
        raise ValueError("No snapshot series matched the selected filters")
    return filtered[0]


def _build_agent_trajectories(
    rows: list[dict[str, str]],
    selected: SnapshotSeries,
    *,
    max_agents: int | None,
) -> dict[int, list[tuple[int, float]]]:
    agent_ids = sorted(
        {
            int(row["agent"])
            for row in rows
            if _row_matches_series(row, selected)
        },
    )
    if max_agents is not None:
        agent_ids = agent_ids[:max_agents]
    selected_agents = set(agent_ids)

    trajectories: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if not _row_matches_series(row, selected):
            continue
        agent = int(row["agent"])
        if agent not in selected_agents:
            continue
        trajectories[agent].append((int(row["step"]), float(row["opinion"])))
    return trajectories


def _row_matches_series(row: dict[str, str], selected: SnapshotSeries) -> bool:
    return (
        row["scenario"] == selected.scenario
        and row["variant"] == selected.variant
        and row["normalizer"] == selected.normalizer
        and row["normalization_every"] == selected.normalization_every
        and row["run_seed"] == selected.run_seed
    )
