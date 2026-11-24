"""Aggregate signed snapshot metrics across Monte Carlo replicas.

Optionally compares a scenario against a control with differences paired by
``run_seed`` (the same seed gives the same initial opinions when both scenarios
come from one experiment).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from atbcr_analysis.metrics import percentile, summarize_values
from atbcr_analysis.plots.signed_metrics import SHIFT_COUNT_COLUMNS, SIGNED_METRIC_COLUMNS

GROUP_COLUMNS = ("scenario", "variant", "normalizer", "normalization_every", "step")
STATISTICS = ("mean", "median", "p10", "p90", "ci95_low", "ci95_high", "n")
_REQUIRED_COLUMNS = {*GROUP_COLUMNS, "run_seed", "opinion_seed"}

# (scenario, variant, normalizer, normalization_every, step)
GroupKey = tuple[str, str, str, str, int]
# (scenario, variant, normalizer, normalization_every, run_seed, step)
RecordKey = tuple[str, str, str, str, str, int]


@dataclass(frozen=True, slots=True)
class DroppedPairs:
    """Replicas of a compared scenario without a matching control record."""

    case: str
    control: str
    runs: int
    snapshots: int


@dataclass(frozen=True, slots=True)
class SignedMetricsSummary:
    """Summary rows ready to be written, plus pairing diagnostics."""

    columns: list[str]
    rows: list[dict[str, object]]
    dropped_pairs: list[DroppedPairs]


def parse_baseline(value: str) -> tuple[str, str]:
    """Parse ``CASE=CONTROL``."""

    case, separator, control = value.partition("=")
    case, control = case.strip(), control.strip()
    if not separator or not case or not control:
        raise ValueError(f"baseline must have the form CASE=CONTROL, got {value!r}")
    if case == control:
        raise ValueError(f"baseline compares {case!r} with itself")
    return case, control


def summarize_signed_metrics(
    signed_metrics_csv: Path,
    *,
    baselines: Sequence[tuple[str, str]] | None = None,
) -> SignedMetricsSummary:
    """Aggregate every signed metric per (scenario, variant, normalizer, every, step)."""

    metric_names, records = _read_records(signed_metrics_csv)
    baseline_map = _baseline_map(baselines or [], records)

    groups: dict[GroupKey, list[tuple[float, ...]]] = {}
    for (scenario, variant, normalizer, every, _run_seed, step), (_seed, values) in (
        records.items()
    ):
        groups.setdefault((scenario, variant, normalizer, every, step), []).append(values)

    deltas: dict[GroupKey, list[tuple[float, ...]]] = {}
    dropped_pairs: list[DroppedPairs] = []
    for case, control in baseline_map.items():
        dropped_runs: set[tuple[str, ...]] = set()
        dropped_snapshots = 0
        for key, (opinion_seed, values) in records.items():
            if key[0] != case:
                continue
            control_record = records.get((control, *key[1:]))
            if control_record is None:
                dropped_runs.add(key[1:5])
                dropped_snapshots += 1
                continue
            control_seed, control_values = control_record
            if control_seed != opinion_seed:
                raise ValueError(
                    f"{case!r} and {control!r} are not paired replicas: run_seed={key[4]} "
                    f"has opinion_seed {opinion_seed} vs {control_seed}",
                )
            deltas.setdefault((case, *key[1:4], key[5]), []).append(
                tuple(a - b for a, b in zip(values, control_values, strict=True)),
            )
        if dropped_snapshots:
            dropped_pairs.append(
                DroppedPairs(
                    case=case,
                    control=control,
                    runs=len(dropped_runs),
                    snapshots=dropped_snapshots,
                ),
            )

    scenario_order: dict[str, int] = {}
    for key in records:
        scenario_order.setdefault(key[0], len(scenario_order))
    ordered_groups = sorted(groups, key=lambda key: (scenario_order[key[0]], *key[1:]))
    columns = [*GROUP_COLUMNS, "runs", *_stat_columns(metric_names)]
    if baseline_map:
        columns += [
            "baseline_scenario",
            "delta_runs",
            *_stat_columns(f"delta_{name}" for name in metric_names),
        ]

    rows: list[dict[str, object]] = []
    for key in ordered_groups:
        replicas = groups[key]
        row: dict[str, object] = dict(zip(GROUP_COLUMNS, key, strict=True))
        row["runs"] = len(replicas)
        row.update(_aggregate(metric_names, replicas))
        if key in deltas:
            paired = deltas[key]
            row["baseline_scenario"] = baseline_map[key[0]]
            row["delta_runs"] = len(paired)
            row.update(_aggregate([f"delta_{name}" for name in metric_names], paired))
        rows.append(row)
    return SignedMetricsSummary(columns=columns, rows=rows, dropped_pairs=dropped_pairs)


def write_signed_metrics_summary_csv(output_path: Path, summary: SignedMetricsSummary) -> None:
    """Write the summary; delta columns stay empty for scenarios without a baseline."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary.columns, restval="")
        writer.writeheader()
        writer.writerows(summary.rows)


def _read_records(
    path: Path,
) -> tuple[list[str], dict[RecordKey, tuple[str, tuple[float, ...]]]]:
    records: dict[RecordKey, tuple[str, tuple[float, ...]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = _REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(
                f"{path}: CSV file is missing required columns: {', '.join(sorted(missing))}",
            )
        metric_names = [
            name
            for name in (*SIGNED_METRIC_COLUMNS, *SHIFT_COUNT_COLUMNS)
            if name in fieldnames
        ]
        if not metric_names:
            raise ValueError(f"{path}: no signed metric columns found")
        for row_number, row in enumerate(reader, start=2):
            try:
                step = int(row["step"])
                values = tuple(_parse_number(row[name]) for name in metric_names)
            except ValueError as error:
                raise ValueError(f"{path}:{row_number}: non-numeric metric value") from error
            key = (
                row["scenario"],
                row["variant"],
                row["normalizer"],
                row["normalization_every"],
                row["run_seed"],
                step,
            )
            if key in records:
                raise ValueError(
                    f"{path}:{row_number}: duplicate row for scenario={key[0]!r}, "
                    f"run_seed={key[4]!r}, step={step}",
                )
            records[key] = (row["opinion_seed"], values)
    if not records:
        raise ValueError(f"{path}: no rows found")
    return metric_names, records


def _parse_number(value: str) -> float:
    return math.nan if value.strip() == "" else float(value)


def _baseline_map(
    baselines: Sequence[tuple[str, str]],
    records: Mapping[RecordKey, object],
) -> dict[str, str]:
    available = {key[0] for key in records}
    mapping: dict[str, str] = {}
    for case, control in baselines:
        for scenario in (case, control):
            if scenario not in available:
                raise ValueError(f"baseline scenario {scenario!r} is not in the input")
        if case in mapping:
            raise ValueError(f"scenario {case!r} has more than one baseline")
        mapping[case] = control
    return mapping


def _stat_columns(metric_names: Iterable[str]) -> list[str]:
    return [f"{name}_{stat}" for name in metric_names for stat in STATISTICS]


def _aggregate(
    metric_names: Sequence[str],
    replicas: Sequence[tuple[float, ...]],
) -> dict[str, float | int]:
    """NaN values are excluded per metric; ``<metric>_n`` counts the valid ones."""

    result: dict[str, float | int] = {}
    for index, name in enumerate(metric_names):
        values = [item[index] for item in replicas if not math.isnan(item[index])]
        if values:
            summary = summarize_values(values)
            stats: dict[str, float | int] = {
                "mean": summary.mean,
                "median": summary.median,
                "p10": percentile(values, 10),
                "p90": percentile(values, 90),
                "ci95_low": summary.ci95_low,
                "ci95_high": summary.ci95_high,
            }
        else:
            stats = dict.fromkeys(STATISTICS[:-1], math.nan)
        stats["n"] = len(values)
        for stat, value in stats.items():
            result[f"{name}_{stat}"] = value
    return result
