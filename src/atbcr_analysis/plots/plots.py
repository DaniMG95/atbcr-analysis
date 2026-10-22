"""Plot helpers for persisted experiment outputs."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple


_GRID_SCENARIO_PATTERN = re.compile(
    r"^e(?P<epsilon>m?\d+(?:p\d+)?)_t(?P<theta>m?\d+(?:p\d+)?)$",
)


class SnapshotSeries(NamedTuple):
    scenario: str
    variant: str
    normalizer: str
    normalization_every: str
    run_seed: str


class ExtremizedShareGrid(NamedTuple):
    """Ordered axes and values for an epsilon/theta heatmap."""

    epsilons: list[float]
    thetas: list[float]
    values: list[list[float]]


def plot_extremized_share_heatmap(
    summary_csv: Path,
    output_path: Path,
    *,
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
) -> None:
    """Plot mean extremized-agent shares for each epsilon/theta combination."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    grid = _read_extremized_share_grid(
        summary_csv,
        variant=variant,
        normalizer=normalizer,
        normalization_every=normalization_every,
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#eeeeee")

    fig, ax = plt.subplots(
        figsize=(
            max(7, len(grid.thetas) * 0.9),
            max(5, len(grid.epsilons) * 0.7),
        ),
    )
    image = ax.imshow(grid.values, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(
        range(len(grid.thetas)),
        labels=[_format_grid_value(value) for value in grid.thetas],
    )
    ax.set_yticks(
        range(len(grid.epsilons)),
        labels=[_format_grid_value(value) for value in grid.epsilons],
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_title("Porcentaje medio de agentes extremizados")

    for row_index, row in enumerate(grid.values):
        for column_index, value in enumerate(row):
            if math.isnan(value):
                continue
            text_color = "white" if value < 0.55 else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:.1%}",
                ha="center",
                va="center",
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, label="Agentes extremizados")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _read_extremized_share_grid(
    summary_csv: Path,
    *,
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
) -> ExtremizedShareGrid:
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or ())
    if not rows:
        raise ValueError("summary file is empty")
    if "extremized_share_mean" not in fieldnames:
        raise ValueError("summary file is missing column 'extremized_share_mean'")

    filters = {
        "variant": variant,
        "normalizer": normalizer,
        "normalization_every": normalization_every,
    }
    selected_rows = [
        row
        for row in rows
        if all(
            expected is None or row.get(column) == expected
            for column, expected in filters.items()
        )
    ]
    if not selected_rows:
        raise ValueError("No summary rows matched the selected filters")

    values_by_pair: dict[tuple[float, float], float] = {}
    for row in selected_rows:
        epsilon, theta = _epsilon_theta_from_summary_row(row)
        share = float(row["extremized_share_mean"])
        if not 0 <= share <= 1:
            raise ValueError("extremized_share_mean values must be between 0 and 1")
        pair = (epsilon, theta)
        if pair in values_by_pair:
            raise ValueError(
                f"Multiple summary rows found for epsilon={epsilon}, theta={theta}; "
                "use the variant/normalizer filters to select one series",
            )
        values_by_pair[pair] = share

    epsilons = sorted({epsilon for epsilon, _ in values_by_pair})
    thetas = sorted({theta for _, theta in values_by_pair})
    values = [
        [values_by_pair.get((epsilon, theta), float("nan")) for theta in thetas]
        for epsilon in epsilons
    ]
    return ExtremizedShareGrid(epsilons, thetas, values)


def _epsilon_theta_from_summary_row(row: dict[str, str]) -> tuple[float, float]:
    if row.get("epsilon", "").strip() and row.get("theta", "").strip():
        return float(row["epsilon"]), float(row["theta"])

    return _epsilon_theta_from_scenario(row.get("scenario", ""))


@lru_cache(maxsize=None)
def _epsilon_theta_from_scenario(scenario: str) -> tuple[float, float]:
    match = _GRID_SCENARIO_PATTERN.fullmatch(scenario)
    if match is None:
        raise ValueError(
            "Cannot determine epsilon and theta: add 'epsilon' and 'theta' columns "
            "or use scenario names like 'e0p1_t0p5'",
        )
    return (
        _parse_grid_number(match.group("epsilon")),
        _parse_grid_number(match.group("theta")),
    )


def _parse_grid_number(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def _format_grid_value(value: float) -> str:
    return f"{value:g}"


def plot_final_opinion_histogram(
    snapshots_csv: Path,
    output_path: Path,
    *,
    epsilon: float | None = None,
    theta: float | None = None,
    parameter_pairs: Sequence[tuple[float, float]] | None = None,
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
    bins: int = 50,
) -> None:
    """Plot final opinions for one or more epsilon/theta parameter pairs."""

    import matplotlib.pyplot as plt

    if bins <= 0:
        raise ValueError("bins must be positive")
    pairs = _resolve_parameter_pairs(
        epsilon=epsilon,
        theta=theta,
        parameter_pairs=parameter_pairs,
    )
    opinions_by_pair = _final_opinions_for_parameter_pairs(
        snapshots_csv,
        parameter_pairs=pairs,
        variant=variant,
        normalizer=normalizer,
        normalization_every=normalization_every,
    )
    all_opinions = [
        opinion for opinions in opinions_by_pair.values() for opinion in opinions
    ]
    lower, upper = _opinion_axis_limits(snapshots_csv, variant, all_opinions)
    bin_edges = [lower + (upper - lower) * index / bins for index in range(bins + 1)]

    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=(max(8, 5 * len(pairs)), 5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, pair in zip(axes[0], pairs, strict=True):
        pair_epsilon, pair_theta = pair
        ax.hist(
            opinions_by_pair[pair],
            bins=bin_edges,
            density=True,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlim(lower, upper)
        ax.set_title(rf"$\epsilon={pair_epsilon:g},\ \theta={pair_theta:g}$")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Distribuciones finales de opinión por configuración")
    fig.supxlabel("Opinión final", y=0.065)
    fig.supylabel("Densidad")
    fig.text(
        0.5,
        0.015,
        "Distribuciones descriptivas: opiniones agregadas del último paso "
        "de cada réplica Monte Carlo.",
        ha="center",
        fontsize="small",
    )
    fig.tight_layout(rect=(0.03, 0.11, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _opinion_axis_limits(
    snapshots_csv: Path,
    variant: str | None,
    opinions: Sequence[float],
) -> tuple[float, float]:
    metadata = _read_run_metadata(snapshots_csv.with_name("config.json"))
    variants = metadata.get("variants", [])
    variant_config = _find_by_name(variants, variant) if variant is not None else {}
    if not variant_config and variant is None and isinstance(variants, list):
        if len(variants) == 1 and isinstance(variants[0], dict):
            variant_config = variants[0]

    domain = variant_config.get("domain", variant)
    if domain == "bounded_01":
        return 0.0, 1.0
    if domain == "bounded_m11":
        return -1.0, 1.0

    lower = min(opinions)
    upper = max(opinions)
    if math.isclose(lower, upper):
        lower -= 0.5
        upper += 0.5
    return lower, upper


def plot_extremized_share_trajectory(
    trajectories_csv: Path,
    output_path: Path,
    *,
    epsilon: float | None = None,
    theta: float | None = None,
    parameter_pairs: Sequence[tuple[float, float]] | None = None,
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
) -> None:
    """Plot mean extremized share over time for one or more parameter pairs."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    pairs = _resolve_parameter_pairs(
        epsilon=epsilon,
        theta=theta,
        parameter_pairs=parameter_pairs,
    )
    trajectories = _mean_extremized_shares_for_parameters(
        trajectories_csv,
        parameter_pairs=pairs,
        variant=variant,
        normalizer=normalizer,
        normalization_every=normalization_every,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for pair in pairs:
        trajectory = trajectories[pair]
        steps = list(trajectory)
        shares = [trajectory[step] for step in steps]
        pair_epsilon, pair_theta = pair
        ax.plot(
            steps,
            shares,
            linewidth=2,
            label=rf"$\epsilon={pair_epsilon:g},\ \theta={pair_theta:g}$",
        )
    ax.set_title("Evolución de agentes extremizados")
    ax.set_xlabel("Paso")
    ax.set_ylabel("Proporción media de agentes extremizados")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _final_opinions_for_parameters(
    snapshots_csv: Path,
    *,
    epsilon: float,
    theta: float,
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
) -> list[float]:
    return _final_opinions_for_parameter_pairs(
        snapshots_csv,
        parameter_pairs=[(epsilon, theta)],
        variant=variant,
        normalizer=normalizer,
        normalization_every=normalization_every,
    )[(epsilon, theta)]


def _final_opinions_for_parameter_pairs(
    snapshots_csv: Path,
    *,
    parameter_pairs: Sequence[tuple[float, float]],
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
) -> dict[tuple[float, float], list[float]]:
    pairs = _resolve_parameter_pairs(parameter_pairs=parameter_pairs)
    final_by_pair_and_run: dict[
        tuple[float, float],
        dict[str, tuple[int, list[float]]],
    ] = {pair: {} for pair in pairs}
    matched_series: dict[tuple[float, float], set[tuple[str, str, str]]] = {
        pair: set() for pair in pairs
    }
    with snapshots_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"run_seed", "step", "opinion"})
        for row in reader:
            pair = _matching_parameter_pair(
                row,
                parameter_pairs=pairs,
                variant=variant,
                normalizer=normalizer,
                normalization_every=normalization_every,
            )
            if pair is None:
                continue
            matched_series[pair].add(_series_key(row))
            run_seed = row["run_seed"]
            step = int(row["step"])
            opinion = float(row["opinion"])
            final_by_run = final_by_pair_and_run[pair]
            current = final_by_run.get(run_seed)
            if current is None or step > current[0]:
                final_by_run[run_seed] = (step, [opinion])
            elif step == current[0]:
                current[1].append(opinion)

    opinions_by_pair: dict[tuple[float, float], list[float]] = {}
    for pair in pairs:
        pair_epsilon, pair_theta = pair
        _validate_parameter_selection(
            matched_series[pair],
            epsilon=pair_epsilon,
            theta=pair_theta,
        )
        opinions_by_pair[pair] = [
            opinion
            for _, opinions in final_by_pair_and_run[pair].values()
            for opinion in opinions
        ]
    return opinions_by_pair


def _mean_extremized_share_for_parameters(
    trajectories_csv: Path,
    *,
    epsilon: float,
    theta: float,
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
) -> dict[int, float]:
    return _mean_extremized_shares_for_parameters(
        trajectories_csv,
        parameter_pairs=[(epsilon, theta)],
        variant=variant,
        normalizer=normalizer,
        normalization_every=normalization_every,
    )[(epsilon, theta)]


def _mean_extremized_shares_for_parameters(
    trajectories_csv: Path,
    *,
    parameter_pairs: Sequence[tuple[float, float]],
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
) -> dict[tuple[float, float], dict[int, float]]:
    pairs = _resolve_parameter_pairs(parameter_pairs=parameter_pairs)
    shares_by_pair_and_step: dict[tuple[float, float], dict[int, list[float]]] = {
        pair: defaultdict(list) for pair in pairs
    }
    matched_series: dict[tuple[float, float], set[tuple[str, str, str]]] = {
        pair: set() for pair in pairs
    }
    with trajectories_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"step", "extremized_share"})
        for row in reader:
            pair = _matching_parameter_pair(
                row,
                parameter_pairs=pairs,
                variant=variant,
                normalizer=normalizer,
                normalization_every=normalization_every,
            )
            if pair is None:
                continue
            matched_series[pair].add(_series_key(row))
            share = float(row["extremized_share"])
            if not 0 <= share <= 1:
                raise ValueError("extremized_share values must be between 0 and 1")
            shares_by_pair_and_step[pair][int(row["step"])].append(share)

    trajectories: dict[tuple[float, float], dict[int, float]] = {}
    for pair in pairs:
        pair_epsilon, pair_theta = pair
        _validate_parameter_selection(
            matched_series[pair],
            epsilon=pair_epsilon,
            theta=pair_theta,
        )
        shares_by_step = shares_by_pair_and_step[pair]
        trajectories[pair] = {
            step: statistics.fmean(shares_by_step[step])
            for step in sorted(shares_by_step)
        }
    return trajectories


def _resolve_parameter_pairs(
    *,
    epsilon: float | None = None,
    theta: float | None = None,
    parameter_pairs: Sequence[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    if (epsilon is None) != (theta is None):
        raise ValueError("epsilon and theta must be provided together")
    pairs = list(parameter_pairs or ())
    if epsilon is not None and theta is not None:
        pairs.append((epsilon, theta))
    if not pairs:
        raise ValueError("at least one epsilon/theta pair is required")
    unique_pairs = list(dict.fromkeys(pairs))
    return unique_pairs


def _matching_parameter_pair(
    row: dict[str, str],
    *,
    parameter_pairs: Sequence[tuple[float, float]],
    variant: str | None,
    normalizer: str | None,
    normalization_every: str | None,
) -> tuple[float, float] | None:
    filters = {
        "variant": variant,
        "normalizer": normalizer,
        "normalization_every": normalization_every,
    }
    if not all(
        expected is None or row.get(column) == expected
        for column, expected in filters.items()
    ):
        return None
    try:
        row_epsilon, row_theta = _epsilon_theta_from_summary_row(row)
    except ValueError:
        return None
    for pair in parameter_pairs:
        epsilon, theta = pair
        if math.isclose(row_epsilon, epsilon) and math.isclose(row_theta, theta):
            return pair
    return None


def _series_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("variant", ""),
        row.get("normalizer", ""),
        row.get("normalization_every", ""),
    )


def _validate_parameter_selection(
    series: set[tuple[str, str, str]],
    *,
    epsilon: float,
    theta: float,
) -> None:
    if not series:
        raise ValueError(
            f"No rows matched epsilon={epsilon}, theta={theta} and the selected filters",
        )
    if len(series) > 1:
        raise ValueError(
            "Multiple series matched the parameter combination; use the "
            "variant/normalizer filters to select one series",
        )


def _require_columns(fieldnames: list[str] | None, required: set[str]) -> None:
    missing = required - set(fieldnames or ())
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"CSV file is missing required columns: {columns}")


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
