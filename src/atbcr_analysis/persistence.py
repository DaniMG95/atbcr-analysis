"""Persistence helpers for experiment results."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from atbcr_analysis.experiments import FINAL_METRIC_NAMES, ExperimentResult


def write_experiment_outputs(
    output_dir: Path,
    results: list[ExperimentResult],
    config: Any,
) -> None:
    """Write summary, trajectories, snapshots, and configuration files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output_dir / "summary.csv", results)
    write_trajectories_csv(output_dir / "trajectories.csv", results)
    write_snapshots_csv(output_dir / "snapshots.csv", results)
    write_normalization_events_csv(output_dir / "normalization_events.csv", results)
    write_json(output_dir / "config.json", _to_jsonable(config))


def write_summary_csv(path: Path, results: list[ExperimentResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scenario",
            "variant",
            "normalizer",
            "normalization_every",
            "runs",
        ]
        for metric_name in FINAL_METRIC_NAMES:
            fieldnames.extend(
                [
                    f"{metric_name}_mean",
                    f"{metric_name}_std",
                    f"{metric_name}_median",
                    f"{metric_name}_ci95_low",
                    f"{metric_name}_ci95_high",
                ],
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "scenario": result.scenario_name,
                "variant": result.variant_name,
                "normalizer": result.normalizer_name,
                "normalization_every": result.normalization_every,
                "runs": result.runs,
            }
            for metric_name, summary in result.metric_summaries.items():
                row[f"{metric_name}_mean"] = summary.mean
                row[f"{metric_name}_std"] = summary.std
                row[f"{metric_name}_median"] = summary.median
                row[f"{metric_name}_ci95_low"] = summary.ci95_low
                row[f"{metric_name}_ci95_high"] = summary.ci95_high
            writer.writerow(row)


def write_trajectories_csv(path: Path, results: list[ExperimentResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scenario",
            "variant",
            "normalizer",
            "normalization_every",
            "run_seed",
            "graph_seed",
            "opinion_seed",
            "dynamics_seed",
            "step",
            "confidence_frequency",
            "inaction_frequency",
            "repulsion_frequency",
            "confidence_frequency_window",
            "inaction_frequency_window",
            "repulsion_frequency_window",
            "max_abs_opinion",
            "mean_abs_opinion",
            "std_opinion",
            "p50_abs_opinion",
            "p90_abs_opinion",
            "p95_abs_opinion",
            "p99_abs_opinion",
            "extremized_share",
            "cluster_count",
            "configured_cluster_tolerance",
            "effective_cluster_tolerance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for simulation in result.simulations:
                for snapshot in simulation.trajectory:
                    writer.writerow(
                        {
                            "scenario": result.scenario_name,
                            "variant": result.variant_name,
                            "normalizer": result.normalizer_name,
                            "normalization_every": result.normalization_every,
                            "run_seed": simulation.seed,
                            "graph_seed": simulation.graph_seed,
                            "opinion_seed": simulation.opinion_seed,
                            "dynamics_seed": simulation.dynamics_seed,
                            **asdict(snapshot.metrics),
                        },
                    )


def write_snapshots_csv(path: Path, results: list[ExperimentResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "variant",
                "normalizer",
                "normalization_every",
                "run_seed",
                "graph_seed",
                "opinion_seed",
                "dynamics_seed",
                "step",
                "agent",
                "opinion",
            ],
        )
        writer.writeheader()
        for result in results:
            for simulation in result.simulations:
                for snapshot in simulation.trajectory:
                    if snapshot.opinions is None:
                        continue
                    for agent, opinion in enumerate(snapshot.opinions):
                        writer.writerow(
                            {
                                "scenario": result.scenario_name,
                                "variant": result.variant_name,
                                "normalizer": result.normalizer_name,
                                "normalization_every": result.normalization_every,
                                "run_seed": simulation.seed,
                                "graph_seed": simulation.graph_seed,
                                "opinion_seed": simulation.opinion_seed,
                                "dynamics_seed": simulation.dynamics_seed,
                                "step": snapshot.metrics.step,
                                "agent": agent,
                                "opinion": opinion,
                            },
                        )


def write_normalization_events_csv(path: Path, results: list[ExperimentResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scenario",
            "variant",
            "normalizer",
            "normalization_every",
            "run_seed",
            "graph_seed",
            "opinion_seed",
            "dynamics_seed",
            "step",
            "max_abs_before",
            "max_abs_after",
            "scale",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for simulation in result.simulations:
                for event in simulation.normalization_events:
                    writer.writerow(
                        {
                            "scenario": result.scenario_name,
                            "variant": result.variant_name,
                            "normalizer": event.normalizer,
                            "normalization_every": result.normalization_every,
                            "run_seed": simulation.seed,
                            "graph_seed": simulation.graph_seed,
                            "opinion_seed": simulation.opinion_seed,
                            "dynamics_seed": simulation.dynamics_seed,
                            "step": event.step,
                            "max_abs_before": event.max_abs_before,
                            "max_abs_after": event.max_abs_after,
                            "scale": event.scale,
                        },
                    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_jsonable(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
