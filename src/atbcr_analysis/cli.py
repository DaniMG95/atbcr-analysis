"""Command line interface for ATBCR experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from atbcr_analysis.config import (
    FileConfig,
    RunConfig,
    SimulationConfig,
)
from atbcr_analysis.experiments import ExperimentResult, run_monte_carlo
from atbcr_analysis.normalizers import NormalizationConfig
from atbcr_analysis.persistence import write_experiment_outputs


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = args.config_path or args.config
    if config_path is None:
        parser.error("a YAML configuration file is required")
    config = FileConfig.from_mapping(_read_yaml_config(config_path))
    for experiment in config.runs:
        _run_experiment(experiment)
    return 0


def _run_experiment(config: RunConfig) -> None:
    scenarios = config.resolved_scenarios
    variants = config.resolved_variants
    normalizations = config.normalizations
    output_dir = config.output_dir

    base_simulation = SimulationConfig(
        name=scenarios[0].name,
        model=scenarios[0].model,
        graph=config.graph,
        variant="baseline_01",
        domain="bounded_01",
        steps=config.steps,
        metrics=config.metrics,
        normalization=NormalizationConfig(),
    )

    results: list[ExperimentResult] = []
    for scenario in scenarios:
        for variant in variants:
            variant_base = base_simulation.with_variant(variant, scenario)
            for normalization in normalizations:
                simulation = _apply_normalization(variant_base, normalization)
                progress = None
                if config.show_progress:
                    progress = _ProgressBar(_progress_label(simulation))
                try:
                    results.append(
                        run_monte_carlo(
                            simulation,
                            config.monte_carlo,
                            progress_callback=progress,
                        ),
                    )
                finally:
                    if progress is not None:
                        progress.finish()

    write_experiment_outputs(
        output_dir,
        results,
        {
            "config": config.model_dump(mode="json"),
            "scenarios": list(scenarios),
            "variants": list(variants),
            "normalizations": list(normalizations),
        },
    )
    _print_summary(results)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atbcr",
        description="Run ATBCR Monte Carlo experiments from a YAML configuration file.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="YAML file with one or more experiment runs.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        help="YAML file with one or more experiment runs.",
    )
    return parser


class _ProgressBar:
    def __init__(self, label: str, *, width: int = 28) -> None:
        self.label = label
        self.width = width
        self.completed = 0
        self.total = 0

    def __call__(self, completed: int, total: int) -> None:
        self.completed = completed
        self.total = total
        ratio = completed / total if total else 1.0
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        remaining = max(total - completed, 0)
        percent = int(ratio * 100)
        sys.stderr.write(
            f"\r{self.label} [{bar}] {completed}/{total} "
            f"({percent:3d}%) quedan {remaining}",
        )
        sys.stderr.flush()

    def finish(self) -> None:
        if self.total == 0:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()


def _progress_label(simulation: SimulationConfig) -> str:
    normalizer = simulation.normalization.kind
    if simulation.normalization.every is not None:
        normalizer = f"{normalizer}/{simulation.normalization.every}"
    return f"{simulation.name}/{simulation.variant}/{normalizer}"


def _read_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        raise ValueError(f"YAML config is empty: {path}")
    if not isinstance(payload, dict):
        raise ValueError("YAML config must be a mapping")
    return payload


def _apply_normalization(
    config: SimulationConfig,
    normalization: NormalizationConfig,
) -> SimulationConfig:
    return SimulationConfig(
        name=config.name,
        model=config.model,
        graph=config.graph,
        variant=config.variant,
        domain=config.domain,
        steps=config.steps,
        metrics=config.metrics,
        opinion_initializer=config.opinion_initializer,
        normalization=normalization,
    )


def _print_summary(results: list[ExperimentResult]) -> None:
    for result in results:
        print(
            f"{result.scenario_name}/{result.variant_name}/"
            f"{result.normalizer_name}/{result.normalization_every}: "
            f"confidence={result.mean_confidence_frequency:.4f}, "
            f"inaction={result.mean_inaction_frequency:.4f}, "
            f"repulsion={result.mean_repulsion_frequency:.4f}, "
            f"max_abs={result.mean_max_abs_opinion:.4f}",
        )


if __name__ == "__main__":
    raise SystemExit(main())
