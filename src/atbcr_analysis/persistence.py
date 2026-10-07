"""Persistence helpers for experiment results."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from atbcr_analysis.experiments import ExperimentResult


def write_experiment_outputs(output_dir: Path, results: list[ExperimentResult], config: Any) -> None:
    """Write summary, trajectories, snapshots, and configuration files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output_dir / "summary.csv", results)
    write_trajectories_csv(output_dir / "trajectories.csv", results)
    write_snapshots_csv(output_dir / "snapshots.csv", results)
    write_json(output_dir / "config.json", _to_jsonable(config))


def write_summary_csv(path: Path, results: list[ExperimentResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scenario",
            "variant",
            "normalizer",
            "normalization_every",
            "runs",
            "confidence_frequency",
            "inaction_frequency",
            "repulsion_frequency",
            "max_abs_opinion",
            "mean_abs_opinion",
            "std_opinion",
            "extremized_share",
            "cluster_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scenario": result.scenario_name,
                    "variant": result.variant_name,
                    "normalizer": result.normalizer_name,
                    "normalization_every": result.normalization_every,
                    "runs": result.runs,
                    "confidence_frequency": result.mean_confidence_frequency,
                    "inaction_frequency": result.mean_inaction_frequency,
                    "repulsion_frequency": result.mean_repulsion_frequency,
                    "max_abs_opinion": result.mean_max_abs_opinion,
                    "mean_abs_opinion": result.mean_abs_opinion,
                    "std_opinion": result.mean_std_opinion,
                    "extremized_share": result.mean_extremized_share,
                    "cluster_count": result.mean_cluster_count,
                },
            )


def write_trajectories_csv(path: Path, results: list[ExperimentResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scenario",
            "variant",
            "normalizer",
            "normalization_every",
            "run_seed",
            "step",
            "confidence_frequency",
            "inaction_frequency",
            "repulsion_frequency",
            "max_abs_opinion",
            "mean_abs_opinion",
            "std_opinion",
            "extremized_share",
            "cluster_count",
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
                                "step": snapshot.metrics.step,
                                "agent": agent,
                                "opinion": opinion,
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
