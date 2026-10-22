"""Generic comparisons of Monte Carlo metric trajectories."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Transform = Literal["none", "log", "log10"]
Aggregate = Literal["mean", "median"]


@dataclass(frozen=True)
class ComparativeSeries:
    """Identity of a series, including its experiment source."""

    source: str
    scenario: str
    variant: str
    normalizer: str
    normalization_every: str
    epsilon: float | None
    theta: float | None


@dataclass(frozen=True)
class AggregatePoint:
    """Aggregate and optional normal-approximation interval at one step."""

    step: int
    value: float
    replicas: int
    ci95_low: float | None
    ci95_high: float | None


@dataclass(frozen=True)
class LinearFit:
    """Ordinary least-squares result for one Monte Carlo replica."""

    series: ComparativeSeries
    run_seed: str
    slope: float
    intercept: float
    r_squared: float
    n_points: int
    fit_start: int
    fit_end: int


ReplicaTrajectories = dict[ComparativeSeries, dict[str, dict[int, float]]]


def read_comparative_trajectories(
    input_paths: Sequence[Path],
    *,
    metric: str,
    transform: Transform = "none",
    parameter_pairs: Collection[tuple[float, float]] | None = None,
    variants: Collection[str] | None = None,
    normalizers: Collection[str] | None = None,
    normalization_every: Collection[str] | None = None,
) -> ReplicaTrajectories:
    """Read and transform metric values while keeping files and replicas separate."""

    if not input_paths:
        raise ValueError("at least one trajectories CSV is required")
    if not metric:
        raise ValueError("metric must not be empty")
    if transform not in {"none", "log", "log10"}:
        raise ValueError(f"unknown transform: {transform}")

    filters = {
        "variant": _optional_set(variants),
        "normalizer": _optional_set(normalizers),
        "normalization_every": _normalization_filter(normalization_every),
    }
    trajectories: ReplicaTrajectories = {}
    unresolved_parameters: set[str] = set()
    for input_path in input_paths:
        scenario_parameters = _read_scenario_parameters(input_path.with_name("config.json"))
        with input_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            _require_columns(
                reader.fieldnames,
                {
                    "scenario",
                    "variant",
                    "normalizer",
                    "normalization_every",
                    "run_seed",
                    "step",
                    metric,
                },
                input_path,
            )
            for row_number, row in enumerate(reader, start=2):
                if not _matches_filters(row, filters):
                    continue
                epsilon, theta = _scenario_parameters(row, scenario_parameters)
                if parameter_pairs and (epsilon is None or theta is None):
                    unresolved_parameters.add(f"{input_path}: {row['scenario']}")
                    continue
                if not _matches_parameter_pairs(epsilon, theta, parameter_pairs):
                    continue
                series = ComparativeSeries(
                    source=str(input_path),
                    scenario=row["scenario"],
                    variant=row["variant"],
                    normalizer=row["normalizer"],
                    normalization_every=row["normalization_every"],
                    epsilon=epsilon,
                    theta=theta,
                )
                try:
                    step = int(row["step"])
                    raw_value = float(row[metric])
                except ValueError as error:
                    raise ValueError(
                        f"{input_path}:{row_number}: metric '{metric}' and step must be numeric",
                    ) from error
                if not math.isfinite(raw_value):
                    raise ValueError(
                        f"{input_path}:{row_number}: metric '{metric}' is not finite",
                    )
                value = _transform_value(
                    raw_value,
                    transform=transform,
                    source=input_path,
                    row_number=row_number,
                    metric=metric,
                )
                run = trajectories.setdefault(series, {}).setdefault(row["run_seed"], {})
                if step in run:
                    raise ValueError(
                        f"{input_path}:{row_number}: duplicate step {step} for run_seed "
                        f"{row['run_seed']} and series {series.scenario}/{series.variant}/"
                        f"{series.normalizer}/{series.normalization_every}",
                    )
                run[step] = value

    if unresolved_parameters:
        unresolved = ", ".join(sorted(unresolved_parameters))
        raise ValueError(
            "cannot resolve epsilon/theta for filtered scenarios; add CSV columns, "
            f"a sibling config.json, or grid-style scenario names: {unresolved}",
        )
    if not trajectories:
        raise ValueError("no trajectory rows matched the selected filters")
    return trajectories


def aggregate_trajectories(
    trajectories: Mapping[ComparativeSeries, Mapping[str, Mapping[int, float]]],
    *,
    aggregate: Aggregate = "mean",
    include_ci: bool = False,
) -> dict[ComparativeSeries, list[AggregatePoint]]:
    """Aggregate transformed values across independent Monte Carlo replicas."""

    if aggregate not in {"mean", "median"}:
        raise ValueError(f"unknown aggregate: {aggregate}")
    result: dict[ComparativeSeries, list[AggregatePoint]] = {}
    for series, runs in trajectories.items():
        values_by_step: dict[int, list[float]] = {}
        for points in runs.values():
            for step, value in points.items():
                values_by_step.setdefault(step, []).append(value)
        result[series] = [
            _aggregate_point(step, values, aggregate=aggregate, include_ci=include_ci)
            for step, values in sorted(values_by_step.items())
        ]
    return result


def fit_replica_trajectories(
    trajectories: Mapping[ComparativeSeries, Mapping[str, Mapping[int, float]]],
    *,
    fit_start: int | None = None,
    fit_end: int | None = None,
) -> list[LinearFit]:
    """Fit each transformed replica independently over the requested interval."""

    if fit_start is not None and fit_end is not None and fit_start > fit_end:
        raise ValueError("fit_start must be less than or equal to fit_end")
    fits: list[LinearFit] = []
    for series, runs in trajectories.items():
        for run_seed, trajectory in runs.items():
            selected = [
                (step, value)
                for step, value in sorted(trajectory.items())
                if (fit_start is None or step >= fit_start)
                and (fit_end is None or step <= fit_end)
            ]
            if len(selected) < 2:
                raise ValueError(
                    f"linear fit needs at least two points for {series.source}, "
                    f"{series.scenario}/{series.variant}/{series.normalizer}/"
                    f"{series.normalization_every}, run_seed={run_seed}",
                )
            x_values = [float(step) for step, _ in selected]
            y_values = [value for _, value in selected]
            slope, intercept, r_squared = _linear_regression(x_values, y_values)
            fits.append(
                LinearFit(
                    series=series,
                    run_seed=run_seed,
                    slope=slope,
                    intercept=intercept,
                    r_squared=r_squared,
                    n_points=len(selected),
                    fit_start=selected[0][0],
                    fit_end=selected[-1][0],
                ),
            )
    return fits


def write_linear_fits_csv(path: Path, fits: Sequence[LinearFit]) -> None:
    """Write replica fits plus one descriptive summary row per series."""

    fieldnames = [
        "record_type",
        "source",
        "scenario",
        "variant",
        "normalizer",
        "normalization_every",
        "epsilon",
        "theta",
        "run_seed",
        "slope",
        "intercept",
        "r_squared",
        "n_points",
        "fit_start",
        "fit_end",
        "replicas",
        "slope_std",
        "slope_ci95_low",
        "slope_ci95_high",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fit in fits:
            writer.writerow(_fit_row(fit))
        for series, series_fits in _fits_by_series(fits).items():
            writer.writerow(_fit_summary_row(series, series_fits))


def plot_comparative_trajectories(
    input_paths: Sequence[Path],
    output_path: Path,
    *,
    metric: str,
    parameter_pairs: Collection[tuple[float, float]] | None = None,
    variants: Collection[str] | None = None,
    normalizers: Collection[str] | None = None,
    normalization_every: Collection[str] | None = None,
    transform: Transform = "none",
    aggregate: Aggregate = "mean",
    include_ci: bool = False,
    fit: bool = False,
    fit_start: int | None = None,
    fit_end: int | None = None,
    fit_output: Path | None = None,
) -> None:
    """Plot generic, filtered trajectory comparisons from one or more CSV files."""

    import matplotlib.pyplot as plt

    trajectories = read_comparative_trajectories(
        input_paths,
        metric=metric,
        transform=transform,
        parameter_pairs=parameter_pairs,
        variants=variants,
        normalizers=normalizers,
        normalization_every=normalization_every,
    )
    aggregates = aggregate_trajectories(
        trajectories,
        aggregate=aggregate,
        include_ci=include_ci,
    )
    fits = fit_replica_trajectories(
        trajectories,
        fit_start=fit_start,
        fit_end=fit_end,
    ) if fit else []
    fits_by_series = _fits_by_series(fits)

    fig, ax = plt.subplots(figsize=(10, 6))
    show_source = len({series.source for series in trajectories}) > 1
    for series, points in aggregates.items():
        series_fits = fits_by_series.get(series, [])
        label = _series_label(series, show_source=show_source)
        if series_fits:
            label += _fit_label(series_fits)
        line = ax.plot(
            [point.step for point in points],
            [point.value for point in points],
            linewidth=2,
            label=label,
        )[0]
        if include_ci:
            ci_points = [point for point in points if point.ci95_low is not None]
            if ci_points:
                ax.fill_between(
                    [point.step for point in ci_points],
                    [point.ci95_low for point in ci_points],
                    [point.ci95_high for point in ci_points],
                    color=line.get_color(),
                    alpha=0.2,
                    linewidth=0,
                )
        if series_fits:
            slope = statistics.fmean(item.slope for item in series_fits)
            intercept = statistics.fmean(item.intercept for item in series_fits)
            lower = min(item.fit_start for item in series_fits)
            upper = max(item.fit_end for item in series_fits)
            ax.plot(
                [lower, upper],
                [intercept + slope * lower, intercept + slope * upper],
                color=line.get_color(),
                linestyle="--",
                linewidth=1.5,
            )

    ax.set_xlabel("step")
    ax.set_ylabel(_metric_axis_label(metric, transform))
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if fit:
        destination = fit_output or output_path.with_name(f"{output_path.stem}_fits.csv")
        write_linear_fits_csv(destination, fits)


def _optional_set(values: Collection[str] | None) -> set[str] | None:
    return set(values) if values else None


def _normalization_filter(values: Collection[str] | None) -> set[str] | None:
    if not values:
        return None
    return {"" if value.lower() in {"none", "null", "-"} else value for value in values}


def _require_columns(
    fieldnames: Sequence[str] | None,
    required: set[str],
    source: Path,
) -> None:
    missing = required - set(fieldnames or ())
    if missing:
        raise ValueError(
            f"{source}: CSV file is missing required columns: {', '.join(sorted(missing))}",
        )


def _matches_filters(
    row: Mapping[str, str],
    filters: Mapping[str, set[str] | None],
) -> bool:
    return all(
        allowed is None or row.get(column, "") in allowed
        for column, allowed in filters.items()
    )


_SCENARIO_PARAMETERS_PATTERN = re.compile(
    r"^e(?P<epsilon>m?\d+(?:p\d+)?)_t(?P<theta>m?\d+(?:p\d+)?)$",
)


def _read_scenario_parameters(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read scenario parameters from {path}: {error}") from error

    parameters: dict[str, tuple[float, float]] = {}
    scenarios = metadata.get("scenarios", []) if isinstance(metadata, dict) else []
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if not isinstance(scenario, dict) or not isinstance(scenario.get("name"), str):
                continue
            model = scenario.get("model", {})
            model_epsilon = model.get("epsilon") if isinstance(model, dict) else None
            epsilon = scenario.get("epsilon", model_epsilon)
            theta = scenario.get("theta", model.get("theta") if isinstance(model, dict) else None)
            if epsilon is not None and theta is not None:
                parameters[scenario["name"]] = (float(epsilon), float(theta))

    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    configured = config.get("scenarios", {}) if isinstance(config, dict) else {}
    if isinstance(configured, dict):
        for name, scenario in configured.items():
            if isinstance(scenario, dict) and "epsilon" in scenario and "theta" in scenario:
                parameters.setdefault(name, (float(scenario["epsilon"]), float(scenario["theta"])))
    return parameters


def _scenario_parameters(
    row: Mapping[str, str],
    configured: Mapping[str, tuple[float, float]],
) -> tuple[float | None, float | None]:
    if row.get("epsilon", "").strip() and row.get("theta", "").strip():
        return float(row["epsilon"]), float(row["theta"])
    scenario = row.get("scenario", "")
    if scenario in configured:
        return configured[scenario]
    match = _SCENARIO_PARAMETERS_PATTERN.fullmatch(scenario)
    if match is None:
        return None, None
    return (
        float(match.group("epsilon").replace("m", "-").replace("p", ".")),
        float(match.group("theta").replace("m", "-").replace("p", ".")),
    )


def _matches_parameter_pairs(
    epsilon: float | None,
    theta: float | None,
    selected: Collection[tuple[float, float]] | None,
) -> bool:
    if not selected:
        return True
    if epsilon is None or theta is None:
        return False
    return any(
        math.isclose(epsilon, selected_epsilon) and math.isclose(theta, selected_theta)
        for selected_epsilon, selected_theta in selected
    )


def _transform_value(
    value: float,
    *,
    transform: Transform,
    source: Path,
    row_number: int,
    metric: str,
) -> float:
    if transform == "none":
        return value
    if value <= 0:
        raise ValueError(
            f"{source}:{row_number}: --transform {transform} requires positive "
            f"'{metric}' values; found {value}",
        )
    return math.log(value) if transform == "log" else math.log10(value)


def _aggregate_point(
    step: int,
    values: Sequence[float],
    *,
    aggregate: Aggregate,
    include_ci: bool,
) -> AggregatePoint:
    center = statistics.fmean(values) if aggregate == "mean" else statistics.median(values)
    low: float | None = None
    high: float | None = None
    if include_ci and len(values) >= 2:
        margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
        low = center - margin
        high = center + margin
    return AggregatePoint(step, center, len(values), low, high)


def _linear_regression(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, float, float]:
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    denominator = sum((value - mean_x) ** 2 for value in x_values)
    if denominator == 0:
        raise ValueError("linear fit requires at least two distinct steps")
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator
    intercept = mean_y - slope * mean_x
    residual_sum = sum(
        (y_value - (intercept + slope * x_value)) ** 2
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    total_sum = sum((value - mean_y) ** 2 for value in y_values)
    r_squared = math.nan if total_sum == 0 else 1 - residual_sum / total_sum
    return slope, intercept, r_squared


def _fits_by_series(fits: Sequence[LinearFit]) -> dict[ComparativeSeries, list[LinearFit]]:
    grouped: dict[ComparativeSeries, list[LinearFit]] = {}
    for fit in fits:
        grouped.setdefault(fit.series, []).append(fit)
    return grouped


def _fit_row(fit: LinearFit) -> dict[str, str | int | float]:
    return {
        "record_type": "replica",
        "source": fit.series.source,
        "scenario": fit.series.scenario,
        "variant": fit.series.variant,
        "normalizer": fit.series.normalizer,
        "normalization_every": fit.series.normalization_every,
        "epsilon": "" if fit.series.epsilon is None else fit.series.epsilon,
        "theta": "" if fit.series.theta is None else fit.series.theta,
        "run_seed": fit.run_seed,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r_squared": fit.r_squared,
        "n_points": fit.n_points,
        "fit_start": fit.fit_start,
        "fit_end": fit.fit_end,
        "replicas": 1,
        "slope_std": "",
        "slope_ci95_low": "",
        "slope_ci95_high": "",
    }


def _fit_summary_row(
    series: ComparativeSeries,
    fits: Sequence[LinearFit],
) -> dict[str, str | int | float]:
    slopes = [fit.slope for fit in fits]
    slope = statistics.fmean(slopes)
    slope_std = statistics.stdev(slopes) if len(slopes) >= 2 else math.nan
    margin = 1.96 * slope_std / math.sqrt(len(slopes)) if len(slopes) >= 2 else math.nan
    finite_r_squared = [fit.r_squared for fit in fits if math.isfinite(fit.r_squared)]
    return {
        "record_type": "summary",
        "source": series.source,
        "scenario": series.scenario,
        "variant": series.variant,
        "normalizer": series.normalizer,
        "normalization_every": series.normalization_every,
        "epsilon": "" if series.epsilon is None else series.epsilon,
        "theta": "" if series.theta is None else series.theta,
        "run_seed": "",
        "slope": slope,
        "intercept": statistics.fmean(fit.intercept for fit in fits),
        "r_squared": statistics.fmean(finite_r_squared) if finite_r_squared else math.nan,
        "n_points": sum(fit.n_points for fit in fits),
        "fit_start": min(fit.fit_start for fit in fits),
        "fit_end": max(fit.fit_end for fit in fits),
        "replicas": len(fits),
        "slope_std": slope_std,
        "slope_ci95_low": slope - margin,
        "slope_ci95_high": slope + margin,
    }


def _series_label(series: ComparativeSeries, *, show_source: bool) -> str:
    frequency = series.normalization_every or "-"
    label = (
        f"{series.scenario}/{series.variant}/{series.normalizer}/every={frequency}"
    )
    if series.epsilon is not None and series.theta is not None:
        label += rf" ($\epsilon={series.epsilon:g}$, $\theta={series.theta:g}$)"
    return f"{series.source}: {label}" if show_source else label


def _fit_label(fits: Sequence[LinearFit]) -> str:
    slope = statistics.fmean(fit.slope for fit in fits)
    finite_r_squared = [fit.r_squared for fit in fits if math.isfinite(fit.r_squared)]
    r_squared = statistics.fmean(finite_r_squared) if finite_r_squared else math.nan
    return rf" ($\bar{{m}}={slope:.4g}$, $\bar{{R}}^2={r_squared:.3f}$)"


def _metric_axis_label(metric: str, transform: Transform) -> str:
    if transform == "log":
        return f"ln({metric})"
    if transform == "log10":
        return f"log10({metric})"
    return metric
