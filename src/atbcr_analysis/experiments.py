"""Monte Carlo experiment orchestration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

from atbcr_analysis.config import MonteCarloConfig, SimulationConfig
from atbcr_analysis.metrics import MetricSummary, summarize_values
from atbcr_analysis.simulation import SimulationResult, SimulationRunner


FINAL_METRIC_NAMES: tuple[str, ...] = (
    "confidence_frequency",
    "inaction_frequency",
    "repulsion_frequency",
    "max_abs_opinion",
    "mean_abs_opinion",
    "std_opinion",
    "extremized_share",
    "cluster_count",
    "p50_abs_opinion",
    "p90_abs_opinion",
    "p95_abs_opinion",
    "p99_abs_opinion",
)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Aggregated Monte Carlo result for one simulation configuration."""

    scenario_name: str
    variant_name: str
    normalizer_name: str
    normalization_every: int | None
    runs: int
    metric_summaries: dict[str, MetricSummary]
    simulations: tuple[SimulationResult, ...]

    @property
    def mean_confidence_frequency(self) -> float:
        return self.metric_summaries["confidence_frequency"].mean

    @property
    def mean_inaction_frequency(self) -> float:
        return self.metric_summaries["inaction_frequency"].mean

    @property
    def mean_repulsion_frequency(self) -> float:
        return self.metric_summaries["repulsion_frequency"].mean

    @property
    def mean_max_abs_opinion(self) -> float:
        return self.metric_summaries["max_abs_opinion"].mean

    @property
    def mean_abs_opinion(self) -> float:
        return self.metric_summaries["mean_abs_opinion"].mean

    @property
    def mean_std_opinion(self) -> float:
        return self.metric_summaries["std_opinion"].mean

    @property
    def mean_extremized_share(self) -> float:
        return self.metric_summaries["extremized_share"].mean

    @property
    def mean_cluster_count(self) -> float:
        return self.metric_summaries["cluster_count"].mean


def run_monte_carlo(
    simulation_config: SimulationConfig,
    monte_carlo_config: MonteCarloConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ExperimentResult:
    """Run independent stochastic simulations and aggregate their outputs."""

    if monte_carlo_config.runs < 1:
        raise ValueError("runs must be positive")
    run_seeds = [
        _run_seed_plan(monte_carlo_config, offset)
        for offset in range(monte_carlo_config.runs)
    ]

    if monte_carlo_config.workers <= 1:
        runner = SimulationRunner.from_config(simulation_config)
        completed_simulations: list[SimulationResult] = []
        for seed_plan in run_seeds:
            completed_simulations.append(
                runner.run(
                    seed_plan.seed,
                    graph_seed=seed_plan.graph_seed,
                    opinion_seed=seed_plan.opinion_seed,
                    dynamics_seed=seed_plan.dynamics_seed,
                ),
            )
            if progress_callback is not None:
                progress_callback(len(completed_simulations), len(run_seeds))
        simulations = tuple(completed_simulations)
    else:
        completed_simulations = []
        with ProcessPoolExecutor(max_workers=monte_carlo_config.workers) as executor:
            for simulation in executor.map(
                _simulate_worker,
                [(simulation_config, seed_plan) for seed_plan in run_seeds],
            ):
                completed_simulations.append(simulation)
                if progress_callback is not None:
                    progress_callback(len(completed_simulations), len(run_seeds))
        simulations = tuple(completed_simulations)

    final_metrics = [run.final_metrics for run in simulations]
    metric_summaries = {
        metric_name: summarize_values(
            [float(getattr(item, metric_name)) for item in final_metrics],
        )
        for metric_name in FINAL_METRIC_NAMES
    }
    return ExperimentResult(
        scenario_name=simulation_config.name,
        variant_name=simulation_config.variant,
        normalizer_name=simulation_config.normalization.kind,
        normalization_every=simulation_config.normalization.every,
        runs=len(simulations),
        metric_summaries=metric_summaries,
        simulations=simulations,
    )


@dataclass(frozen=True, slots=True)
class RunSeedPlan:
    """Master and optional phase-specific seeds for one Monte Carlo run."""

    seed: int
    graph_seed: int | None
    opinion_seed: int | None
    dynamics_seed: int | None


def _run_seed_plan(config: MonteCarloConfig, offset: int) -> RunSeedPlan:
    return RunSeedPlan(
        seed=config.seed + offset,
        graph_seed=None if config.graph_seed is None else config.graph_seed + offset,
        opinion_seed=None if config.opinion_seed is None else config.opinion_seed + offset,
        dynamics_seed=None if config.dynamics_seed is None else config.dynamics_seed + offset,
    )


def _simulate_worker(args: tuple[SimulationConfig, RunSeedPlan]) -> SimulationResult:
    config, seed_plan = args
    return SimulationRunner.from_config(config).run(
        seed_plan.seed,
        graph_seed=seed_plan.graph_seed,
        opinion_seed=seed_plan.opinion_seed,
        dynamics_seed=seed_plan.dynamics_seed,
    )
