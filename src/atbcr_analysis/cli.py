"""Command line interface for ATBCR experiments."""

from __future__ import annotations

import argparse
import sys
import time
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
    if config.show_progress:
        _print_status(
            f"Experimento iniciado: {len(scenarios)} escenarios, "
            f"{len(variants)} variantes y {len(normalizations)} normalizaciones.",
        )
    for scenario_number, scenario in enumerate(scenarios, start=1):
        for variant in variants:
            variant_base = base_simulation.with_variant(variant, scenario)
            for normalization in normalizations:
                simulation = _apply_normalization(variant_base, normalization)
                progress = None
                if config.show_progress:
                    _print_status(
                        f"Escenario {scenario_number}/{len(scenarios)}: "
                        f"preparando {_progress_label(simulation)}...",
                    )
                    progress = _ProgressBar(_progress_label(simulation))
                    progress.start(config.monte_carlo.runs, simulation.steps)
                try:
                    results.append(
                        run_monte_carlo(
                            simulation,
                            config.monte_carlo,
                            step_progress_callback=(
                                None if progress is None else progress.update_steps
                            ),
                        ),
                    )
                finally:
                    if progress is not None:
                        progress.finish()
                if config.show_progress:
                    _print_status(
                        f"Escenario {scenario_number}/{len(scenarios)} completado.",
                    )

    if config.show_progress:
        _print_status(f"Guardando resultados en {output_dir}...")
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
    if config.show_progress:
        _print_status("Resultados guardados. Experimento completado.")
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
        self.started_at = time.monotonic()

    def start(self, total_runs: int, total_steps: int) -> None:
        self.update_steps(1, total_runs, 0, total_steps)

    def __call__(self, completed: int, total: int) -> None:
        self.completed = completed
        self.total = total
        ratio = completed / total if total else 1.0
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        remaining = max(total - completed, 0)
        elapsed = time.monotonic() - self.started_at
        remaining_seconds = elapsed / completed * remaining if completed else 0.0
        percent = int(ratio * 100)
        sys.stderr.write(
            f"\r{self.label} [{bar}] {completed}/{total} "
            f"({percent:3d}%) quedan {remaining} | ETA {_format_duration(remaining_seconds)}",
        )
        sys.stderr.flush()

    def update_steps(
        self,
        run_number: int,
        total_runs: int,
        step: int,
        total_steps: int,
    ) -> None:
        completed_steps = (run_number - 1) * total_steps + step
        all_steps = total_runs * total_steps
        ratio = completed_steps / all_steps if all_steps else 1.0
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self.started_at
        eta = _estimate_remaining(elapsed, completed_steps, all_steps)
        percent = ratio * 100
        sys.stderr.write(
            f"\r{self.label} [{bar}] replica {run_number}/{total_runs} | "
            f"paso {step}/{total_steps} | total {percent:5.1f}% | "
            f"transcurrido {_format_duration(elapsed)} | ETA {eta}",
        )
        sys.stderr.flush()
        self.completed = completed_steps
        self.total = all_steps

    def finish(self) -> None:
        if self.total == 0:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _estimate_remaining(elapsed: float, completed: int, total: int) -> str:
    if completed <= 0:
        return "calculando..."
    remaining_seconds = elapsed / completed * max(total - completed, 0)
    return _format_duration(remaining_seconds)


def _print_status(message: str) -> None:
    sys.stderr.write(f"{message}\n")
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
