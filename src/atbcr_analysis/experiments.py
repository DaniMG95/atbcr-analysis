"""Monte Carlo experiment orchestration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

from atbcr_analysis.config import MonteCarloConfig, SimulationConfig
from atbcr_analysis.metrics import mean
from atbcr_analysis.simulation import SimulationResult, SimulationRunner


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Aggregated Monte Carlo result for one simulation configuration."""

    scenario_name: str
    variant_name: str
    normalizer_name: str
    normalization_every: int | None
    runs: int
    mean_confidence_frequency: float
    mean_inaction_frequency: float
    mean_repulsion_frequency: float
    mean_max_abs_opinion: float
    mean_abs_opinion: float
    mean_std_opinion: float
    mean_extremized_share: float
    mean_cluster_count: float
    simulations: tuple[SimulationResult, ...]


def run_monte_carlo(
    simulation_config: SimulationConfig,
    monte_carlo_config: MonteCarloConfig,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ExperimentResult:
    """Run independent stochastic simulations and aggregate their outputs."""

    if monte_carlo_config.runs < 1:
        raise ValueError("runs must be positive")
    seeds = [monte_carlo_config.seed + offset for offset in range(monte_carlo_config.runs)]

    if monte_carlo_config.workers <= 1:
        runner = SimulationRunner.from_config(simulation_config)
        completed_simulations: list[SimulationResult] = []
        for seed in seeds:
            completed_simulations.append(runner.run(seed))
            if progress_callback is not None:
                progress_callback(len(completed_simulations), len(seeds))
        simulations = tuple(completed_simulations)
    else:
        completed_simulations = []
        with ProcessPoolExecutor(max_workers=monte_carlo_config.workers) as executor:
            for simulation in executor.map(
                _simulate_worker,
                [(simulation_config, seed) for seed in seeds],
            ):
                completed_simulations.append(simulation)
                if progress_callback is not None:
                    progress_callback(len(completed_simulations), len(seeds))
        simulations = tuple(completed_simulations)

    final_metrics = [run.final_metrics for run in simulations]
    return ExperimentResult(
        scenario_name=simulation_config.name,
        variant_name=simulation_config.variant,
        normalizer_name=simulation_config.normalization.kind,
        normalization_every=simulation_config.normalization.every,
        runs=len(simulations),
        mean_confidence_frequency=mean([item.confidence_frequency for item in final_metrics]),
        mean_inaction_frequency=mean([item.inaction_frequency for item in final_metrics]),
        mean_repulsion_frequency=mean([item.repulsion_frequency for item in final_metrics]),
        mean_max_abs_opinion=mean([item.max_abs_opinion for item in final_metrics]),
        mean_abs_opinion=mean([item.mean_abs_opinion for item in final_metrics]),
        mean_std_opinion=mean([item.std_opinion for item in final_metrics]),
        mean_extremized_share=mean([item.extremized_share for item in final_metrics]),
        mean_cluster_count=mean([float(item.cluster_count) for item in final_metrics]),
        simulations=simulations,
    )


def _simulate_worker(args: tuple[SimulationConfig, int]) -> SimulationResult:
    config, seed = args
    return SimulationRunner.from_config(config).run(seed)
