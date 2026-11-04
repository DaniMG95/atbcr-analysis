"""Temporal opinion stability derived from consecutive persisted snapshots."""

from __future__ import annotations

import csv
import json
import math
import mmap
import statistics
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice, product
from pathlib import Path
from typing import cast


_SEED_COLUMNS = ("run_seed", "graph_seed", "opinion_seed", "dynamics_seed")
_CACHE_VERSION = 1
_MMAP_SEARCH_BLOCK_SIZE = 64 * 1024 * 1024
_CACHE_FIELDNAMES = [
    "source",
    "scenario",
    "variant",
    "normalizer",
    "normalization_every",
    *_SEED_COLUMNS,
    "previous_step",
    "step",
    "max_opinion_change",
]

ProgressCallback = Callable[[Path, int, int], None]
StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class StabilitySeries:
    """Identity of one experiment series, including its source file."""

    source: str
    scenario: str
    variant: str
    normalizer: str
    normalization_every: str


@dataclass(frozen=True, order=True)
class SnapshotInterval:
    """Two consecutive stored snapshot steps in one Monte Carlo run."""

    previous_step: int
    step: int


@dataclass(frozen=True)
class StabilityPoint:
    """Mean maximum opinion change for one snapshot interval."""

    previous_step: int
    step: int
    value: float
    replicas: int


RunKey = tuple[str, str, str, str]
RunStability = dict[StabilitySeries, dict[RunKey, dict[SnapshotInterval, float]]]


def read_snapshot_stability(
    input_paths: Sequence[Path],
    *,
    scenarios: Collection[str] | None = None,
    variants: Collection[str] | None = None,
    normalizers: Collection[str] | None = None,
    normalization_every: Collection[str] | None = None,
    run_seed: str | None = None,
    chunk_size: int = 10_000,
    progress: ProgressCallback | None = None,
) -> RunStability:
    """Calculate maximum changes between consecutive snapshots, run by run."""

    if not input_paths:
        raise ValueError("at least one snapshots CSV is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    filters = {
        "scenario": _optional_set(scenarios),
        "variant": _optional_set(variants),
        "normalizer": _optional_set(normalizers),
        "normalization_every": _normalization_filter(normalization_every),
        "run_seed": {run_seed} if run_seed is not None else None,
    }
    stability: RunStability = {}
    for input_path in input_paths:
        if not _read_exact_runs_mmap(
            input_path,
            filters=filters,
            destination=stability,
            progress=progress,
        ):
            _read_snapshot_file(
                input_path,
                filters=filters,
                chunk_size=chunk_size,
                destination=stability,
                progress=progress,
            )
    if not stability:
        raise ValueError("no snapshot rows matched the selected filters")
    if not any(runs for runs in stability.values()):
        raise ValueError("matched runs need at least two snapshots to calculate stability")
    if run_seed is not None and any(len(runs) != 1 for runs in stability.values()):
        raise ValueError(
            f"run_seed={run_seed} did not identify exactly one run per configuration",
        )
    return stability


def aggregate_snapshot_stability(
    stability: Mapping[
        StabilitySeries,
        Mapping[RunKey, Mapping[SnapshotInterval, float]],
    ],
) -> dict[StabilitySeries, list[StabilityPoint]]:
    """Average per-run changes over replicas for each exact snapshot interval."""

    result: dict[StabilitySeries, list[StabilityPoint]] = {}
    for series, runs in stability.items():
        values_by_interval: dict[SnapshotInterval, list[float]] = {}
        for intervals in runs.values():
            for interval, value in intervals.items():
                values_by_interval.setdefault(interval, []).append(value)
        if values_by_interval:
            result[series] = [
                StabilityPoint(
                    previous_step=interval.previous_step,
                    step=interval.step,
                    value=statistics.fmean(values),
                    replicas=len(values),
                )
                for interval, values in sorted(values_by_interval.items())
            ]
    return result


def write_snapshot_stability_csv(path: Path, stability: RunStability) -> None:
    """Atomically persist compact per-run, per-interval stability values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=_CACHE_FIELDNAMES)
            writer.writeheader()
            for series, runs in stability.items():
                for run_key, intervals in runs.items():
                    seeds = dict(zip(_SEED_COLUMNS, run_key, strict=True))
                    for interval, value in sorted(intervals.items()):
                        writer.writerow(
                            {
                                "source": series.source,
                                "scenario": series.scenario,
                                "variant": series.variant,
                                "normalizer": series.normalizer,
                                "normalization_every": series.normalization_every,
                                **seeds,
                                "previous_step": interval.previous_step,
                                "step": interval.step,
                                "max_opinion_change": value,
                            },
                        )
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_snapshot_stability_csv(path: Path) -> RunStability:
    """Read compact stability values previously derived from snapshots."""

    stability: RunStability = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, set(_CACHE_FIELDNAMES), path)
        for row_number, row in enumerate(reader, start=2):
            try:
                interval = SnapshotInterval(int(row["previous_step"]), int(row["step"]))
                value = float(row["max_opinion_change"])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{row_number}: cached steps and change must be numeric",
                ) from error
            if interval.previous_step >= interval.step:
                raise ValueError(
                    f"{path}:{row_number}: cached snapshot interval must be increasing",
                )
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{path}:{row_number}: cached change must be finite and non-negative",
                )
            series = StabilitySeries(
                source=row["source"],
                scenario=row["scenario"],
                variant=row["variant"],
                normalizer=row["normalizer"],
                normalization_every=row["normalization_every"],
            )
            run_key = cast(RunKey, tuple(row[column] for column in _SEED_COLUMNS))
            run = stability.setdefault(series, {}).setdefault(run_key, {})
            if interval in run:
                raise ValueError(f"{path}:{row_number}: duplicate cached interval for one run")
            run[interval] = value
    if not stability:
        raise ValueError("stability cache is empty")
    return stability


def plot_snapshot_stability(
    input_paths: Sequence[Path],
    output_path: Path,
    *,
    scenarios: Collection[str] | None = None,
    variants: Collection[str] | None = None,
    normalizers: Collection[str] | None = None,
    normalization_every: Collection[str] | None = None,
    run_seed: str | None = None,
    chunk_size: int = 10_000,
    cache_path: Path | None = None,
    refresh_cache: bool = False,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> None:
    """Plot maximum changes between consecutive stored opinion snapshots."""

    import matplotlib.pyplot as plt

    selection = _selection_signature(
        scenarios=scenarios,
        variants=variants,
        normalizers=normalizers,
        normalization_every=normalization_every,
        run_seed=run_seed,
    )
    if cache_path is not None and not refresh_cache and _cache_is_current(
        cache_path,
        input_paths,
        selection,
    ):
        if status is not None:
            status(f"Reutilizando cache de estabilidad: {cache_path}")
        stability = read_snapshot_stability_csv(cache_path)
    else:
        if cache_path is not None and status is not None:
            status(f"Generando cache de estabilidad: {cache_path}")
        stability = read_snapshot_stability(
            input_paths,
            scenarios=scenarios,
            variants=variants,
            normalizers=normalizers,
            normalization_every=normalization_every,
            run_seed=run_seed,
            chunk_size=chunk_size,
            progress=progress,
        )
        if status is not None:
            status("Datos de snapshots procesados; generando la grafica.")
        if cache_path is not None:
            write_snapshot_stability_csv(cache_path, stability)
            _write_cache_metadata(cache_path, input_paths, selection)

    aggregates = aggregate_snapshot_stability(stability)
    fig, ax = plt.subplots(figsize=(10, 6))
    show_source = len({series.source for series in aggregates}) > 1
    for series, points in aggregates.items():
        ax.plot(
            [point.step for point in points],
            [point.value for point in points],
            marker="o",
            markersize=3,
            linewidth=2,
            label=_series_label(series, show_source=show_source),
        )
    ax.set_title("Estabilidad temporal entre snapshots consecutivos")
    ax.set_xlabel("Paso del snapshot final del intervalo")
    if run_seed is None:
        ax.set_ylabel(r"Media de $\Delta_{\max}$ entre ejecuciones")
    else:
        ax.set_ylabel(rf"$\Delta_{{\max}}$ para run_seed={run_seed}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.text(
        0.5,
        0.01,
        "Cada punto compara dos snapshots almacenados consecutivos; "
        "no representa necesariamente un solo paso de simulacion.",
        ha="center",
        fontsize="small",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read_snapshot_file(
    input_path: Path,
    *,
    filters: Mapping[str, set[str] | None],
    chunk_size: int,
    destination: RunStability,
    progress: ProgressCallback | None,
) -> None:
    total_bytes = input_path.stat().st_size
    if progress is not None:
        progress(input_path, 0, total_bytes)
    processed_bytes = 0
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        fieldnames = next(reader, None)
        required = {
            "scenario",
            "variant",
            "normalizer",
            "normalization_every",
            "run_seed",
            "step",
            "agent",
            "opinion",
        }
        _require_columns(fieldnames, required, input_path)
        indices = {name: index for index, name in enumerate(fieldnames or ())}
        expected_series = _expected_series(input_path, filters)
        completed_series: set[StabilitySeries] = set()
        completed_runs: set[tuple[StabilitySeries, RunKey]] = set()
        current_identity: tuple[StabilitySeries, RunKey] | None = None
        current_step: int | None = None
        current_opinions: dict[int, float] = {}
        previous_step: int | None = None
        previous_opinions: dict[int, float] | None = None
        row_number = 1

        def flush_snapshot() -> None:
            nonlocal previous_step, previous_opinions
            if current_identity is None or current_step is None:
                return
            if previous_step is not None and previous_opinions is not None:
                _validate_same_agents(
                    previous_opinions,
                    current_opinions,
                    source=input_path,
                    previous_step=previous_step,
                    step=current_step,
                )
                series, run_key = current_identity
                interval = SnapshotInterval(previous_step, current_step)
                run = destination.setdefault(series, {}).setdefault(run_key, {})
                run[interval] = max(
                    abs(current_opinions[agent] - previous_opinions[agent])
                    for agent in current_opinions
                )
            previous_step = current_step
            previous_opinions = current_opinions.copy()

        stop_early = False
        while not stop_early:
            rows = list(islice(reader, chunk_size))
            if not rows:
                break
            for row in rows:
                row_number += 1
                series = StabilitySeries(
                    source=str(input_path),
                    scenario=row[indices["scenario"]],
                    variant=row[indices["variant"]],
                    normalizer=row[indices["normalizer"]],
                    normalization_every=row[indices["normalization_every"]],
                )
                run_key = _run_key(row, indices)
                identity = (series, run_key)
                if current_identity is not None and identity != current_identity:
                    flush_snapshot()
                    completed_runs.add(current_identity)
                    completed_series.add(current_identity[0])
                    current_identity = None
                    current_step = None
                    current_opinions = {}
                    previous_step = None
                    previous_opinions = None
                    if expected_series is not None and expected_series <= completed_series:
                        stop_early = True
                        break
                if not _sequence_row_matches_filters(row, indices, filters):
                    continue
                if identity in completed_runs:
                    raise ValueError(
                        f"{input_path}:{row_number}: rows for each run must be contiguous",
                    )
                if current_identity is None:
                    current_identity = identity
                    current_step = _parse_int(row, indices["step"], "step", input_path, row_number)
                    current_opinions = {}
                    destination.setdefault(series, {})
                step = _parse_int(row, indices["step"], "step", input_path, row_number)
                agent = _parse_int(row, indices["agent"], "agent", input_path, row_number)
                opinion = _parse_float(
                    row,
                    indices["opinion"],
                    "opinion",
                    input_path,
                    row_number,
                )
                if current_step is not None and step != current_step:
                    if step < current_step:
                        raise ValueError(
                            f"{input_path}:{row_number}: snapshot steps must be ordered "
                            "within each run",
                        )
                    flush_snapshot()
                    current_step = step
                    current_opinions = {}
                if agent in current_opinions:
                    raise ValueError(
                        f"{input_path}:{row_number}: duplicate agent {agent} at step {step}",
                    )
                current_opinions[agent] = opinion
            if progress is not None:
                processed_bytes = min(handle.buffer.tell(), total_bytes)
                progress(input_path, processed_bytes, total_bytes)
        flush_snapshot()
    if progress is not None:
        progress(input_path, processed_bytes if stop_early else total_bytes, total_bytes)


def _read_exact_runs_mmap(
    input_path: Path,
    *,
    filters: Mapping[str, set[str] | None],
    destination: RunStability,
    progress: ProgressCallback | None,
) -> bool:
    filter_columns = ("scenario", "variant", "normalizer", "normalization_every", "run_seed")
    if any(filters[column] is None for column in filter_columns):
        return False
    with input_path.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
        fieldnames = header.decode("utf-8").split(",")
        if fieldnames[:5] != list(filter_columns):
            return False
        required = {*filter_columns, *_SEED_COLUMNS, "step", "agent", "opinion"}
        _require_columns(fieldnames, required, input_path)
        indices = {name: index for index, name in enumerate(fieldnames)}
        total_bytes = input_path.stat().st_size
        if progress is not None:
            progress(input_path, 0, total_bytes)
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            cursor = len(header) + 1
            for scenario, variant, normalizer, frequency in product(
                sorted(filters["scenario"] or ()),
                sorted(filters["variant"] or ()),
                sorted(filters["normalizer"] or ()),
                sorted(filters["normalization_every"] or (), key=_frequency_sort_key),
            ):
                run_seed = next(iter(filters["run_seed"] or ()))
                values = (scenario, variant, normalizer, frequency, run_seed)
                if any("," in value or "\n" in value or "\r" in value for value in values):
                    return False
                prefix = ",".join(values).encode("utf-8") + b","
                position = _find_line_prefix(
                    mapped,
                    prefix,
                    start=cursor,
                    total_bytes=total_bytes,
                    input_path=input_path,
                    progress=progress,
                )
                if position is None and cursor > len(header) + 1:
                    position = _find_line_prefix(
                        mapped,
                        prefix,
                        start=len(header) + 1,
                        total_bytes=total_bytes,
                        input_path=input_path,
                        progress=progress,
                    )
                if position is None:
                    joined = "/".join(values)
                    raise ValueError(f"{input_path}: selected run was not found: {joined}")
                cursor = _consume_mapped_run(
                    mapped,
                    position=position,
                    prefix=prefix,
                    indices=indices,
                    input_path=input_path,
                    destination=destination,
                )
                if progress is not None:
                    progress(input_path, min(cursor, total_bytes), total_bytes)
    return True


def _find_line_prefix(
    mapped: mmap.mmap,
    prefix: bytes,
    *,
    start: int,
    total_bytes: int,
    input_path: Path,
    progress: ProgressCallback | None,
) -> int | None:
    pattern = b"\n" + prefix
    search_start = max(0, start - 1)
    while search_start < total_bytes:
        search_end = min(total_bytes, search_start + _MMAP_SEARCH_BLOCK_SIZE)
        position = mapped.find(pattern, search_start, search_end)
        if position >= 0:
            return position + 1
        if progress is not None:
            progress(input_path, search_end, total_bytes)
        if search_end == total_bytes:
            break
        search_start = search_end - len(pattern) + 1
    return None


def _consume_mapped_run(
    mapped: mmap.mmap,
    *,
    position: int,
    prefix: bytes,
    indices: Mapping[str, int],
    input_path: Path,
    destination: RunStability,
) -> int:
    previous_step: int | None = None
    previous_opinions: dict[int, float] | None = None
    current_step: int | None = None
    current_opinions: dict[int, float] = {}
    series: StabilitySeries | None = None
    run_key: RunKey | None = None

    def flush_snapshot() -> None:
        nonlocal previous_step, previous_opinions
        if current_step is None:
            return
        if previous_step is not None and previous_opinions is not None:
            _validate_same_agents(
                previous_opinions,
                current_opinions,
                source=input_path,
                previous_step=previous_step,
                step=current_step,
            )
            if series is None or run_key is None:
                raise RuntimeError("mapped run identity was not initialized")
            interval = SnapshotInterval(previous_step, current_step)
            destination.setdefault(series, {}).setdefault(run_key, {})[interval] = max(
                abs(current_opinions[agent] - previous_opinions[agent])
                for agent in current_opinions
            )
        previous_step = current_step
        previous_opinions = current_opinions.copy()

    row_number = 0
    while position < len(mapped) and mapped[position : position + len(prefix)] == prefix:
        line_end = mapped.find(b"\n", position)
        if line_end < 0:
            line_end = len(mapped)
        fields = mapped[position:line_end].rstrip(b"\r").split(b",")
        row_number += 1
        if series is None:
            series = StabilitySeries(
                source=str(input_path),
                scenario=fields[indices["scenario"]].decode("utf-8"),
                variant=fields[indices["variant"]].decode("utf-8"),
                normalizer=fields[indices["normalizer"]].decode("utf-8"),
                normalization_every=fields[indices["normalization_every"]].decode("utf-8"),
            )
            run_key = cast(
                RunKey,
                tuple(fields[indices[column]].decode("utf-8") for column in _SEED_COLUMNS),
            )
        try:
            step = int(fields[indices["step"]])
            agent = int(fields[indices["agent"]])
            opinion = float(fields[indices["opinion"]])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"{input_path}: malformed selected snapshot row {row_number}",
            ) from error
        if not math.isfinite(opinion):
            raise ValueError(f"{input_path}: selected snapshot opinion must be finite")
        if current_step is None:
            current_step = step
        elif step != current_step:
            if step < current_step:
                raise ValueError(f"{input_path}: selected snapshot steps are not ordered")
            flush_snapshot()
            current_step = step
            current_opinions = {}
        if agent in current_opinions:
            raise ValueError(f"{input_path}: duplicate agent {agent} at step {step}")
        current_opinions[agent] = opinion
        position = line_end + 1
    flush_snapshot()
    return position


def _frequency_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _expected_series(
    source: Path,
    filters: Mapping[str, set[str] | None],
) -> set[StabilitySeries] | None:
    keys = ("scenario", "variant", "normalizer", "normalization_every", "run_seed")
    if any(filters[key] is None for key in keys):
        return None
    return {
        StabilitySeries(str(source), scenario, variant, normalizer, frequency)
        for scenario, variant, normalizer, frequency in product(
            filters["scenario"] or (),
            filters["variant"] or (),
            filters["normalizer"] or (),
            filters["normalization_every"] or (),
        )
    }


def _run_key(row: Sequence[str], indices: Mapping[str, int]) -> RunKey:
    values = tuple(row[indices[column]] if column in indices else "" for column in _SEED_COLUMNS)
    return cast(RunKey, values)


def _parse_int(
    row: Sequence[str],
    index: int,
    column: str,
    source: Path,
    row_number: int,
) -> int:
    try:
        return int(row[index])
    except (IndexError, ValueError) as error:
        raise ValueError(f"{source}:{row_number}: {column} must be an integer") from error


def _parse_float(
    row: Sequence[str],
    index: int,
    column: str,
    source: Path,
    row_number: int,
) -> float:
    try:
        value = float(row[index])
    except (IndexError, ValueError) as error:
        raise ValueError(f"{source}:{row_number}: {column} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{source}:{row_number}: {column} must be finite")
    return value


def _validate_same_agents(
    previous: Mapping[int, float],
    current: Mapping[int, float],
    *,
    source: Path,
    previous_step: int,
    step: int,
) -> None:
    previous_agents = set(previous)
    current_agents = set(current)
    if previous_agents != current_agents:
        missing = sorted(previous_agents - current_agents)
        added = sorted(current_agents - previous_agents)
        raise ValueError(
            f"{source}: agent sets differ between snapshots {previous_step} and {step}; "
            f"missing={missing}, added={added}",
        )


def _selection_signature(
    *,
    scenarios: Collection[str] | None,
    variants: Collection[str] | None,
    normalizers: Collection[str] | None,
    normalization_every: Collection[str] | None,
    run_seed: str | None,
) -> dict[str, list[str] | str | None]:
    return {
        "scenarios": sorted(scenarios) if scenarios else None,
        "variants": sorted(variants) if variants else None,
        "normalizers": sorted(normalizers) if normalizers else None,
        "normalization_every": (
            sorted(_normalization_filter(normalization_every) or ())
            if normalization_every
            else None
        ),
        "run_seed": run_seed,
    }


def _cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.meta.json")


def _source_fingerprints(input_paths: Sequence[Path]) -> list[dict[str, str | int]]:
    fingerprints: list[dict[str, str | int]] = []
    for path in input_paths:
        stat = path.stat()
        fingerprints.append(
            {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        )
    return fingerprints


def _cache_is_current(
    cache_path: Path,
    input_paths: Sequence[Path],
    selection: Mapping[str, object],
) -> bool:
    metadata_path = _cache_metadata_path(cache_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        sources = _source_fingerprints(input_paths)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and metadata.get("version") == _CACHE_VERSION
        and metadata.get("sources") == sources
        and metadata.get("selection") == selection
    )


def _write_cache_metadata(
    cache_path: Path,
    input_paths: Sequence[Path],
    selection: Mapping[str, object],
) -> None:
    metadata_path = _cache_metadata_path(cache_path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                {
                    "version": _CACHE_VERSION,
                    "sources": _source_fingerprints(input_paths),
                    "selection": selection,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        temporary_path.replace(metadata_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _optional_set(values: Collection[str] | None) -> set[str] | None:
    return set(values) if values else None


def _normalization_filter(values: Collection[str] | None) -> set[str] | None:
    if not values:
        return None
    return {"" if value.lower() in {"none", "null", "-"} else value for value in values}


def _sequence_row_matches_filters(
    row: Sequence[str],
    indices: Mapping[str, int],
    filters: Mapping[str, set[str] | None],
) -> bool:
    return all(
        allowed is None or row[indices[column]] in allowed
        for column, allowed in filters.items()
    )


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


def _series_label(series: StabilitySeries, *, show_source: bool) -> str:
    frequency = series.normalization_every or "-"
    label = f"{series.scenario}/{series.variant}/{series.normalizer}/every={frequency}"
    return f"{series.source}: {label}" if show_source else label
