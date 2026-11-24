"""Signed per-snapshot opinion metrics derived from ``snapshots.csv``.

The metrics persisted in ``trajectories.csv`` are mostly absolute values, which
hide whether a ``bounded_m11`` distribution drifts towards +1 or -1. This module
streams ``snapshots.csv`` one contiguous (run, step) block at a time and keeps
the sign.
"""

from __future__ import annotations

import csv
import math
import tempfile
from collections.abc import Collection, Iterable, Iterator
from dataclasses import astuple, dataclass
from pathlib import Path

from atbcr_analysis.metrics import mean, percentile, population_std

IDENTITY_COLUMNS = (
    "scenario",
    "variant",
    "normalizer",
    "normalization_every",
    "run_seed",
    "graph_seed",
    "opinion_seed",
    "dynamics_seed",
)
SHIFT_COUNT_COLUMNS = ("atbcr_steps", "shift_steps")
SIGNED_METRIC_COLUMNS = (
    "agent_count",
    "mean_opinion",
    "median_opinion",
    "q10_opinion",
    "q25_opinion",
    "q75_opinion",
    "q90_opinion",
    "iqr_opinion",
    "width_q90_q10",
    "std_opinion",
    "extreme_pos_share",
    "extreme_neg_share",
    "at_pos_boundary_share",
    "at_neg_boundary_share",
    "central_count",
    "central_mean",
    "central_median",
    "central_std",
)
_REQUIRED_SNAPSHOT_COLUMNS = {*IDENTITY_COLUMNS, "step", "opinion"}
OUTPUT_COLUMNS = [*IDENTITY_COLUMNS, "step", *SIGNED_METRIC_COLUMNS, *SHIFT_COUNT_COLUMNS]

DEFAULT_EXTREME_CUTOFF = 0.8
DEFAULT_BOUNDARY_TOL = 1e-12

SnapshotKey = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SignedOpinionMetrics:
    """Signed distribution metrics for one set of opinions."""

    agent_count: int
    mean_opinion: float
    median_opinion: float
    q10_opinion: float
    q25_opinion: float
    q75_opinion: float
    q90_opinion: float
    iqr_opinion: float
    width_q90_q10: float
    std_opinion: float
    extreme_pos_share: float
    extreme_neg_share: float
    at_pos_boundary_share: float
    at_neg_boundary_share: float
    central_count: int
    central_mean: float
    central_median: float
    central_std: float


@dataclass(frozen=True, slots=True)
class SignedSnapshotMetrics:
    """Signed metrics for one Monte Carlo run at one stored step."""

    scenario: str
    variant: str
    normalizer: str
    normalization_every: str
    run_seed: str
    graph_seed: str
    opinion_seed: str
    dynamics_seed: str
    step: int
    metrics: SignedOpinionMetrics
    atbcr_steps: str = ""
    shift_steps: str = ""


def signed_opinion_metrics(
    opinions: list[float],
    *,
    extreme_cutoff: float = DEFAULT_EXTREME_CUTOFF,
    boundary_tol: float = DEFAULT_BOUNDARY_TOL,
) -> SignedOpinionMetrics:
    """Compute signed metrics, reusing ``metrics.percentile`` and ``population_std``.

    ``opinions`` must be in agent order so ``std_opinion`` reproduces the value
    stored in ``trajectories.csv`` exactly (floating-point sums depend on order).
    """

    if not opinions:
        raise ValueError("opinions cannot be empty")
    count = len(opinions)
    ordered = sorted(opinions)
    q10 = percentile(ordered, 10)
    q25 = percentile(ordered, 25)
    q75 = percentile(ordered, 75)
    q90 = percentile(ordered, 90)
    upper_boundary = 1 - boundary_tol
    lower_boundary = -1 + boundary_tol
    central = [opinion for opinion in opinions if abs(opinion) < extreme_cutoff]
    return SignedOpinionMetrics(
        agent_count=count,
        mean_opinion=mean(opinions),
        median_opinion=percentile(ordered, 50),
        q10_opinion=q10,
        q25_opinion=q25,
        q75_opinion=q75,
        q90_opinion=q90,
        iqr_opinion=q75 - q25,
        width_q90_q10=q90 - q10,
        std_opinion=population_std(opinions),
        extreme_pos_share=sum(opinion >= extreme_cutoff for opinion in opinions) / count,
        extreme_neg_share=sum(opinion <= -extreme_cutoff for opinion in opinions) / count,
        at_pos_boundary_share=sum(opinion >= upper_boundary for opinion in opinions) / count,
        at_neg_boundary_share=sum(opinion <= lower_boundary for opinion in opinions) / count,
        central_count=len(central),
        central_mean=mean(central) if central else math.nan,
        central_median=percentile(central, 50) if central else math.nan,
        central_std=population_std(central) if central else math.nan,
    )


def calculate_signed_snapshot_metrics(
    snapshots_csv: Path,
    *,
    trajectories_csv: Path | None = None,
    extreme_cutoff: float = DEFAULT_EXTREME_CUTOFF,
    boundary_tol: float = DEFAULT_BOUNDARY_TOL,
    scenarios: Collection[str] | None = None,
) -> Iterator[SignedSnapshotMetrics]:
    """Yield one record per contiguous (identity, step) block of ``snapshots_csv``.

    Only one snapshot is held in memory. A block key that reappears after its
    block was closed raises ``ValueError`` because the file is not ordered as
    ``persistence`` writes it.
    """

    if extreme_cutoff <= 0:
        raise ValueError("extreme_cutoff must be positive")
    if boundary_tol < 0:
        raise ValueError("boundary_tol must be non-negative")
    scenario_filter = set(scenarios) if scenarios else None
    shift_counts = (
        _read_shift_counts(trajectories_csv, scenario_filter)
        if trajectories_csv is not None
        else None
    )
    return _iter_signed_snapshot_metrics(
        snapshots_csv,
        shift_counts=shift_counts,
        extreme_cutoff=extreme_cutoff,
        boundary_tol=boundary_tol,
        scenario_filter=scenario_filter,
    )


def write_signed_snapshot_metrics_csv(
    output_path: Path,
    metrics: Iterable[SignedSnapshotMetrics],
) -> int:
    """Write records as they are produced; the file only appears if all succeed."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    written = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(OUTPUT_COLUMNS)
            for item in metrics:
                writer.writerow(
                    [
                        item.scenario,
                        item.variant,
                        item.normalizer,
                        item.normalization_every,
                        item.run_seed,
                        item.graph_seed,
                        item.opinion_seed,
                        item.dynamics_seed,
                        item.step,
                        *astuple(item.metrics),
                        item.atbcr_steps,
                        item.shift_steps,
                    ],
                )
                written += 1
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return written


def _iter_signed_snapshot_metrics(
    snapshots_csv: Path,
    *,
    shift_counts: dict[SnapshotKey, tuple[str, str]] | None,
    extreme_cutoff: float,
    boundary_tol: float,
    scenario_filter: set[str] | None,
) -> Iterator[SignedSnapshotMetrics]:
    completed_keys: set[SnapshotKey] = set()
    current_key: SnapshotKey | None = None
    current_opinions: list[float] = []
    emitted = 0

    def build(key: SnapshotKey, opinions: list[float]) -> SignedSnapshotMetrics:
        try:
            step = int(key[-1])
        except ValueError as error:
            raise ValueError(f"{snapshots_csv}: step {key[-1]!r} is not an integer") from error
        counts = ("", "")
        if shift_counts is not None:
            if key not in shift_counts:
                raise ValueError(
                    f"no trajectories row for scenario={key[0]!r}, run_seed={key[4]!r}, "
                    f"step={key[-1]}",
                )
            counts = shift_counts[key]
        return SignedSnapshotMetrics(
            *key[:-1],
            step=step,
            metrics=signed_opinion_metrics(
                opinions,
                extreme_cutoff=extreme_cutoff,
                boundary_tol=boundary_tol,
            ),
            atbcr_steps=counts[0],
            shift_steps=counts[1],
        )

    with snapshots_csv.open(newline="", encoding="utf-8") as handle:
        header_line = handle.readline()
        fieldnames = next(csv.reader([header_line]), [])
        missing = _REQUIRED_SNAPSHOT_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(
                f"{snapshots_csv}: CSV file is missing required columns: "
                f"{', '.join(sorted(missing))}",
            )
        key_indices = [fieldnames.index(column) for column in (*IDENTITY_COLUMNS, "step")]
        scenario_index = key_indices[0]
        opinion_index = fieldnames.index("opinion")
        lines: Iterable[str] = handle
        if scenario_filter is not None and scenario_index == 0:
            lines = _prefilter_lines(handle, scenario_filter)

        for row in csv.reader(lines):
            if not row:
                continue
            if scenario_filter is not None and row[scenario_index] not in scenario_filter:
                continue
            key = tuple(row[index] for index in key_indices)
            if key != current_key:
                if current_key is not None:
                    yield build(current_key, current_opinions)
                    emitted += 1
                    completed_keys.add(current_key)
                if key in completed_keys:
                    raise ValueError(
                        f"{snapshots_csv}: snapshot block scenario={key[0]!r}, "
                        f"run_seed={key[4]!r}, step={key[-1]} reappears after it was "
                        "closed; rows must be contiguous per run and step",
                    )
                current_key = key
                current_opinions = []
            try:
                opinion = float(row[opinion_index])
            except ValueError as error:
                raise ValueError(
                    f"{snapshots_csv}: opinion {row[opinion_index]!r} is not numeric",
                ) from error
            if not math.isfinite(opinion):
                raise ValueError(f"{snapshots_csv}: opinion must be finite")
            current_opinions.append(opinion)

    if current_key is not None:
        yield build(current_key, current_opinions)
        emitted += 1
    if emitted == 0:
        raise ValueError(f"{snapshots_csv}: no snapshot rows matched the selected scenarios")


def _prefilter_lines(lines: Iterable[str], scenarios: set[str]) -> Iterator[str]:
    """Drop lines of other scenarios before CSV parsing (the main cost on big files).

    Quoted first fields are passed through and filtered after parsing.
    """

    prefixes = tuple(f"{scenario}," for scenario in scenarios)
    for line in lines:
        if line.startswith(prefixes) or line.startswith('"'):
            yield line


def _read_shift_counts(
    trajectories_csv: Path,
    scenario_filter: set[str] | None,
) -> dict[SnapshotKey, tuple[str, str]]:
    counts: dict[SnapshotKey, tuple[str, str]] = {}
    with trajectories_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = {*IDENTITY_COLUMNS, "step"} - fieldnames
        if missing:
            raise ValueError(
                f"{trajectories_csv}: CSV file is missing required columns: "
                f"{', '.join(sorted(missing))}",
            )
        has_shift_columns = set(SHIFT_COUNT_COLUMNS) <= fieldnames
        for row in reader:
            if scenario_filter is not None and row["scenario"] not in scenario_filter:
                continue
            key = (*(row[column] for column in IDENTITY_COLUMNS), row["step"])
            counts[key] = (
                (row["atbcr_steps"], row["shift_steps"]) if has_shift_columns else ("", "")
            )
    return counts

