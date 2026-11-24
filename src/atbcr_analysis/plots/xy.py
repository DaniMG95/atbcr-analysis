"""Mean of one trajectories column against another, one line per scenario."""

from __future__ import annotations

import csv
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from atbcr_analysis.plots.comparative import (
    _matches_filters,
    _normalization_filter,
    _optional_set,
    _require_columns,
)


@dataclass(frozen=True)
class XYSeries:
    """Identity of one plotted line."""

    source: str
    scenario: str
    variant: str
    normalizer: str
    normalization_every: str


@dataclass(frozen=True)
class XYPoint:
    """Mean of ``y`` over every row that shares the same ``x`` (or x bin)."""

    x: float
    y_mean: float
    count: int


def read_xy_means(
    input_paths: Sequence[Path],
    *,
    x: str,
    y: str,
    run_seeds: Collection[str] | None = None,
    scenarios: Collection[str] | None = None,
    variants: Collection[str] | None = None,
    normalizers: Collection[str] | None = None,
    normalization_every: Collection[str] | None = None,
    bins: int | None = None,
) -> dict[XYSeries, list[XYPoint]]:
    """Group rows by series and by ``x`` value, averaging ``y`` in each group.

    With ``bins`` the ``x`` axis of each series is split into equal-width bins
    and each point is placed at the mean ``x`` of the rows inside its bin.
    """

    if not input_paths:
        raise ValueError("at least one trajectories CSV is required")
    if bins is not None and bins < 1:
        raise ValueError("bins must be a positive integer")

    filters = {
        "run_seed": _optional_set(run_seeds),
        "scenario": _optional_set(scenarios),
        "variant": _optional_set(variants),
        "normalizer": _optional_set(normalizers),
        "normalization_every": _normalization_filter(normalization_every),
    }
    raw: dict[XYSeries, list[tuple[float, float]]] = {}
    for input_path in input_paths:
        with input_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            _require_columns(
                reader.fieldnames,
                {"scenario", "variant", "normalizer", "normalization_every", "run_seed", x, y},
                input_path,
            )
            for row_number, row in enumerate(reader, start=2):
                if not _matches_filters(row, filters):
                    continue
                try:
                    x_value = float(row[x])
                    y_value = float(row[y])
                except ValueError as error:
                    raise ValueError(
                        f"{input_path}:{row_number}: columns '{x}' and '{y}' must be numeric",
                    ) from error
                if not (math.isfinite(x_value) and math.isfinite(y_value)):
                    continue
                series = XYSeries(
                    source=str(input_path),
                    scenario=row["scenario"],
                    variant=row["variant"],
                    normalizer=row["normalizer"],
                    normalization_every=row["normalization_every"],
                )
                raw.setdefault(series, []).append((x_value, y_value))

    if not raw:
        raise ValueError("no rows matched the requested filters")
    return {
        series: _bin_points(pairs, bins) if bins else _group_points(pairs)
        for series, pairs in raw.items()
    }


def _group_points(pairs: list[tuple[float, float]]) -> list[XYPoint]:
    groups: dict[float, list[float]] = {}
    for x_value, y_value in pairs:
        groups.setdefault(x_value, []).append(y_value)
    return [
        XYPoint(x=x_value, y_mean=sum(ys) / len(ys), count=len(ys))
        for x_value, ys in sorted(groups.items())
    ]


def _bin_points(pairs: list[tuple[float, float]], bins: int) -> list[XYPoint]:
    x_min = min(x_value for x_value, _ in pairs)
    x_max = max(x_value for x_value, _ in pairs)
    width = (x_max - x_min) / bins
    groups: dict[int, list[tuple[float, float]]] = {}
    for x_value, y_value in pairs:
        index = 0 if width == 0 else min(int((x_value - x_min) / width), bins - 1)
        groups.setdefault(index, []).append((x_value, y_value))
    return [
        XYPoint(
            x=sum(p[0] for p in members) / len(members),
            y_mean=sum(p[1] for p in members) / len(members),
            count=len(members),
        )
        for _, members in sorted(groups.items())
    ]


def plot_xy_means(
    input_paths: Sequence[Path],
    output_path: Path,
    *,
    x: str,
    y: str,
    run_seeds: Collection[str] | None = None,
    scenarios: Collection[str] | None = None,
    variants: Collection[str] | None = None,
    normalizers: Collection[str] | None = None,
    normalization_every: Collection[str] | None = None,
    bins: int | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    markers: bool = False,
) -> dict[XYSeries, list[XYPoint]]:
    """Plot mean ``y`` against ``x`` with one line per scenario/variant series."""

    import matplotlib.pyplot as plt

    data = read_xy_means(
        input_paths,
        x=x,
        y=y,
        run_seeds=run_seeds,
        scenarios=scenarios,
        variants=variants,
        normalizers=normalizers,
        normalization_every=normalization_every,
        bins=bins,
    )
    label_fields = _varying_label_fields(data)

    ordered = sorted(data.items(), key=lambda item: _sort_key(item[0]))
    # tab10 repeats after 10 lines; switch to tab20 and then add line styles.
    palette = plt.get_cmap("tab10" if len(ordered) <= 10 else "tab20").colors
    line_styles = ("-", "--", ":", "-.")
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, (series, points) in enumerate(ordered):
        ax.plot(
            [p.x for p in points],
            [p.y_mean for p in points],
            marker="o" if markers else None,
            markersize=3,
            linewidth=1.5,
            color=palette[index % len(palette)],
            linestyle=line_styles[(index // len(palette)) % len(line_styles)],
            label=_series_label(series, label_fields),
        )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(x)
    ax.set_ylabel(f"mean {y}")
    ax.set_title(_title(x, y, run_seeds))
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return data


def _sort_key(series: XYSeries) -> tuple[str, ...]:
    return (
        series.source,
        series.scenario,
        series.variant,
        series.normalizer,
        series.normalization_every,
    )


def _varying_label_fields(data: dict[XYSeries, list[XYPoint]]) -> list[str]:
    fields = ["source", "scenario", "variant", "normalizer", "normalization_every"]
    varying = [
        field
        for field in fields
        if len({getattr(series, field) for series in data}) > 1
    ]
    return varying or ["scenario"]


def _series_label(series: XYSeries, fields: list[str]) -> str:
    parts = []
    for field in fields:
        value = getattr(series, field)
        if field == "source":
            value = Path(value).parent.name or value
        elif field == "normalization_every":
            value = f"every={value or '-'}"
        parts.append(value)
    return " | ".join(parts)


def _title(x: str, y: str, run_seeds: Collection[str] | None) -> str:
    if run_seeds:
        seeds = ", ".join(sorted(run_seeds, key=lambda s: (len(s), s)))
        label = "seed" if len(run_seeds) == 1 else "seeds"
        return f"mean {y} vs {x} ({label} {seeds})"
    return f"mean {y} vs {x} (all seeds)"
