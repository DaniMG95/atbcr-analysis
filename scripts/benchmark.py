"""Basic performance benchmark for complete-graph ATBCR runs."""

from __future__ import annotations

import argparse
import time

from atbcr_analysis.config import (
    ATBCRModelConfig,
    CompleteGraphConfig,
    MetricsConfig,
    MonteCarloConfig,
    SimulationConfig,
)
from atbcr_analysis.experiments import run_monte_carlo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure approximate ATBCR complete-graph throughput.",
    )
    parser.add_argument("--agents", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--runs", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    simulation = SimulationConfig(
        name="benchmark",
        model=ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.1),
        graph=CompleteGraphConfig(n_agents=args.agents),
        variant="baseline_01",
        domain="bounded_01",
        steps=args.steps,
        metrics=MetricsConfig(
            record_every=args.steps,
            store_opinion_snapshots=False,
            store_interaction_events=False,
        ),
    )

    print("runs,agents,steps,total_seconds,seconds_per_run,interactions_per_second")
    for runs in args.runs:
        start = time.perf_counter()
        run_monte_carlo(
            simulation,
            MonteCarloConfig(runs=runs, seed=args.seed, workers=1),
        )
        elapsed = time.perf_counter() - start
        interactions = runs * args.steps
        print(
            f"{runs},{args.agents},{args.steps},{elapsed:.6f},"
            f"{elapsed / runs:.6f},{interactions / elapsed:.2f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
