"""Per-snapshot opinion metrics derived from persisted agent opinions."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


_SCENARIO_PATTERN = re.compile(
    r"^e(?P<epsilon>m?\d+(?:p\d+)?)_t(?P<theta>m?\d+(?:p\d+)?)$",
)
_IDENTITY_COLUMNS = (
    "scenario",
    "variant",
    "normalizer",
    "normalization_every",
    "run_seed",
    "graph_seed",
    "opinion_seed",
    "dynamics_seed",
)
_REQUIRED_COLUMNS = {
    "scenario",
    "variant",
    "normalizer",
    "normalization_every",
    "run_seed",
    "step",
    "agent",
    "opinion",
}
_OUTPUT_COLUMNS = [
    *_IDENTITY_COLUMNS,
    "step",
    "agent_count",
    "proportion_abs_lt_1",
    "closest_to_zero_agent",
    "min_abs_opinion",
    "closest_pair_agent_i",
    "closest_pair_agent_j",
    "min_pairwise_distance",
    "confidence_epsilon",
    "confidence_pair_exists",
]


@dataclass(frozen=True, slots=True)
class SnapshotOpinionMetrics:
    """Three opinion-state metrics for one Monte Carlo snapshot."""

    scenario: str
    variant: str
    normalizer: str
    normalization_every: str
    run_seed: str
    graph_seed: str
    opinion_seed: str
    dynamics_seed: str
    step: int
    agent_count: int
    proportion_abs_lt_1: float
    closest_to_zero_agent: int
    min_abs_opinion: float
    closest_pair_agent_i: int | None
    closest_pair_agent_j: int | None
    min_pairwise_distance: float | None
    confidence_epsilon: float
    confidence_pair_exists: bool


SnapshotKey = tuple[str, str, str, str, str, str, str, str, int]


def calculate_snapshot_opinion_metrics(
    snapshots_csv: Path,
    *,
    epsilon: float,
    theta: float,
    variant: str | None = None,
    normalizer: str | None = None,
    normalization_every: str | None = None,
    run_seed: str | None = None,
) -> list[SnapshotOpinionMetrics]:
    """Calculate metrics per matching run and step while streaming the CSV."""

    if epsilon < 0 or theta < 0:
        raise ValueError("epsilon and theta must be non-negative")
    if epsilon >= theta:
        raise ValueError("epsilon must be strictly smaller than theta")
    normalization_filter = _normalization_filter(normalizer, normalization_every)
    exact_run_selection = all(
        value is not None
        for value in (variant, normalizer, normalization_filter, run_seed)
    )
    threshold_scales = _read_threshold_scales(snapshots_csv.with_name("config.json"))
    metrics: list[SnapshotOpinionMetrics] = []
    selected_series: set[tuple[str, str, str]] = set()
    completed_keys: set[SnapshotKey] = set()
    current_key: SnapshotKey | None = None
    current_agents: set[int] = set()
    current_opinions: list[tuple[int, float]] = []
    matched_run_started = False

    def flush_snapshot() -> None:
        nonlocal current_agents, current_opinions
        if current_key is None:
            return
        metrics.append(
            _summarize_snapshot(
                current_key,
                current_opinions,
                epsilon=epsilon,
                threshold_scale=threshold_scales.get(current_key[1], 1.0),
            ),
        )
        completed_keys.add(current_key)
        current_agents = set()
        current_opinions = []

    with snapshots_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, snapshots_csv)
        for row_number, row in enumerate(reader, start=2):
            matches = _row_matches(
                row,
                epsilon=epsilon,
                theta=theta,
                variant=variant,
                normalizer=normalizer,
                normalization_every=normalization_filter,
                run_seed=run_seed,
            )
            if not matches:
                if matched_run_started and exact_run_selection:
                    break
                continue
            matched_run_started = True
            key = _snapshot_key(row, snapshots_csv, row_number)
            if key != current_key:
                flush_snapshot()
                if key in completed_keys:
                    raise ValueError(
                        f"{snapshots_csv}:{row_number}: snapshot rows are not contiguous",
                    )
                current_key = key
            try:
                agent = int(row["agent"])
                opinion = float(row["opinion"])
            except ValueError as error:
                raise ValueError(
                    f"{snapshots_csv}:{row_number}: agent and opinion must be numeric",
                ) from error
            if not math.isfinite(opinion):
                raise ValueError(
                    f"{snapshots_csv}:{row_number}: opinion must be finite",
                )
            if agent in current_agents:
                raise ValueError(
                    f"{snapshots_csv}:{row_number}: duplicate agent {agent} at step {key[-1]}",
                )
            current_agents.add(agent)
            current_opinions.append((agent, opinion))
            selected_series.add((row["variant"], row["normalizer"], row["normalization_every"]))
    flush_snapshot()

    if not metrics:
        raise ValueError(
            f"No rows matched epsilon={epsilon}, theta={theta} and the selected filters",
        )
    if len(selected_series) > 1:
        raise ValueError(
            "Multiple series matched; use --variant, --normalizer and "
            "--normalization-every to select one",
        )
    return metrics


def write_snapshot_opinion_metrics_csv(
    output_path: Path,
    metrics: list[SnapshotOpinionMetrics],
) -> None:
    """Write compact per-snapshot metrics to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_OUTPUT_COLUMNS)
        writer.writeheader()
        for item in metrics:
            row = asdict(item)
            if item.min_pairwise_distance is None:
                row["min_pairwise_distance"] = ""
            writer.writerow(row)


def _summarize_snapshot(
    key: SnapshotKey,
    opinions: list[tuple[int, float]],
    *,
    epsilon: float,
    threshold_scale: float,
) -> SnapshotOpinionMetrics:
    if not opinions:
        raise ValueError("a snapshot must contain at least one opinion")
    closest_agent, closest_opinion = min(
        opinions,
        key=lambda item: (abs(item[1]), item[0]),
    )
    sorted_opinions = sorted(opinions, key=lambda item: (item[1], item[0]))
    closest_pair: tuple[float, int, int] | None = None
    if len(sorted_opinions) > 1:
        pair_candidates = (
            (right[1] - left[1], min(left[0], right[0]), max(left[0], right[0]))
            for left, right in zip(sorted_opinions, sorted_opinions[1:])
        )
        closest_pair = min(pair_candidates)
    min_pairwise_distance = closest_pair[0] if closest_pair is not None else None
    confidence_epsilon = epsilon * threshold_scale
    identity = key[:-1]
    return SnapshotOpinionMetrics(
        *identity,
        step=key[-1],
        agent_count=len(opinions),
        proportion_abs_lt_1=(
            sum(abs(opinion) < 1 for _, opinion in opinions) / len(opinions)
        ),
        closest_to_zero_agent=closest_agent,
        min_abs_opinion=abs(closest_opinion),
        closest_pair_agent_i=closest_pair[1] if closest_pair is not None else None,
        closest_pair_agent_j=closest_pair[2] if closest_pair is not None else None,
        min_pairwise_distance=min_pairwise_distance,
        confidence_epsilon=confidence_epsilon,
        confidence_pair_exists=(
            min_pairwise_distance is not None
            and min_pairwise_distance < confidence_epsilon
        ),
    )


def _snapshot_key(row: dict[str, str], source: Path, row_number: int) -> SnapshotKey:
    try:
        step = int(row["step"])
    except ValueError as error:
        raise ValueError(f"{source}:{row_number}: step must be an integer") from error
    identity = tuple(row.get(column, "") for column in _IDENTITY_COLUMNS)
    return (*identity, step)


def _row_matches(
    row: dict[str, str],
    *,
    epsilon: float,
    theta: float,
    variant: str | None,
    normalizer: str | None,
    normalization_every: str | None,
    run_seed: str | None,
) -> bool:
    filters = {
        "variant": variant,
        "normalizer": normalizer,
        "normalization_every": normalization_every,
        "run_seed": run_seed,
    }
    if not all(
        expected is None or row.get(column) == expected
        for column, expected in filters.items()
    ):
        return False
    try:
        row_epsilon, row_theta = _scenario_parameters(row)
    except ValueError:
        return False
    return math.isclose(row_epsilon, epsilon) and math.isclose(row_theta, theta)


def _normalization_filter(
    normalizer: str | None,
    normalization_every: str | None,
) -> str | None:
    if normalization_every is not None:
        if normalization_every.lower() in {"none", "null", "-"}:
            return ""
        return normalization_every
    if normalizer == "none":
        return ""
    return None


def _scenario_parameters(row: dict[str, str]) -> tuple[float, float]:
    if row.get("epsilon", "").strip() and row.get("theta", "").strip():
        return float(row["epsilon"]), float(row["theta"])
    return _parameters_from_scenario(row.get("scenario", ""))


@lru_cache(maxsize=None)
def _parameters_from_scenario(scenario: str) -> tuple[float, float]:
    match = _SCENARIO_PATTERN.fullmatch(scenario)
    if match is None:
        raise ValueError(f"cannot determine epsilon and theta from scenario {scenario!r}")
    return (
        _parse_scenario_number(match.group("epsilon")),
        _parse_scenario_number(match.group("theta")),
    )


def _parse_scenario_number(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def _read_threshold_scales(config_path: Path) -> dict[str, float]:
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    variants = payload.get("variants", []) if isinstance(payload, dict) else []
    return {
        str(item["name"]): float(item.get("threshold_scale", 1.0))
        for item in variants
        if isinstance(item, dict) and "name" in item
    }


def _require_columns(fieldnames: list[str] | None, source: Path) -> None:
    missing = _REQUIRED_COLUMNS - set(fieldnames or ())
    if missing:
        raise ValueError(
            f"{source}: CSV file is missing required columns: {', '.join(sorted(missing))}",
        )
