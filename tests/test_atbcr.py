from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from atbcr_analysis import plot_cli
from atbcr_analysis.config import (
    ATBCRModelConfig,
    BarabasiAlbertGraphConfig,
    ErdosRenyiGraphConfig,
    MetricsConfig,
    ModelConfig,
    MonteCarloConfig,
    RingGraphConfig,
    SimulationConfig,
    UniformOpinionInitializerConfig,
)
from atbcr_analysis.experiments import run_monte_carlo
from atbcr_analysis.graphs import generate_graph, graph_edges
from atbcr_analysis.metrics import extremized_share
from atbcr_analysis.models import register_model
from atbcr_analysis.simulation import SimulationRunner, simulate_atbcr


class GraphGenerationTests(unittest.TestCase):
    def test_barabasi_albert_generates_edges(self) -> None:
        import random

        graph = generate_graph(BarabasiAlbertGraphConfig(n_agents=20, ba_m=2), random.Random(1))

        self.assertEqual(len(graph), 20)
        self.assertGreater(len(graph_edges(graph)), 0)


class SimulationTests(unittest.TestCase):
    def test_extremized_share_uses_both_edges_for_unit_interval(self) -> None:
        opinions = [0.01, 0.04, 0.50, 0.96, 0.99]

        share = extremized_share(opinions, threshold=0.9, domain="bounded_01")

        self.assertEqual(share, 4 / 5)

    def test_extremized_share_uses_absolute_value_for_centered_domains(self) -> None:
        opinions = [-0.95, -0.20, 0.20, 0.95]

        share = extremized_share(opinions, threshold=0.9, domain="bounded_m11")

        self.assertEqual(share, 2 / 4)

    def test_simulation_keeps_opinions_in_unit_interval(self) -> None:
        config = SimulationConfig(
            name="test",
            model=ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.3),
            graph=RingGraphConfig(n_agents=30),
            steps=200,
            metrics=MetricsConfig(record_every=50),
        )

        result = simulate_atbcr(config, seed=123)

        self.assertEqual(result.seed, 123)
        self.assertTrue(all(0 <= opinion <= 1 for opinion in result.final_opinions))
        self.assertEqual(result.trajectory[0].metrics.step, 0)
        self.assertEqual(result.trajectory[-1].metrics.step, 200)

    def test_uniform_initializer_accepts_custom_ranges(self) -> None:
        import random

        initializer = UniformOpinionInitializerConfig(low=0.2, high=0.4).build()
        opinions = initializer.initialize(20, random.Random(1))

        self.assertTrue(all(0.2 <= opinion <= 0.4 for opinion in opinions))

        config = SimulationConfig(
            name="test",
            model=ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.3),
            graph=RingGraphConfig(n_agents=30),
            steps=1,
            metrics=MetricsConfig(record_every=1),
            opinion_initializer=UniformOpinionInitializerConfig(low=0.2, high=0.4),
        )

        result = SimulationRunner.from_config(config).run(seed=123)

        self.assertTrue(all(0 <= opinion <= 1 for opinion in result.final_opinions))
        self.assertTrue(any(0.2 <= opinion <= 0.4 for opinion in result.final_opinions))

    def test_runner_can_use_registered_model(self) -> None:
        class FrozenModel:
            name = "frozen"

            def step(self, opinions, adjacency, edges, rng) -> str:
                return "inaction"

            def validate(self) -> None:
                return None

        register_model("frozen", lambda model_config, domain: FrozenModel())
        config = SimulationConfig(
            name="unused",
            model=ModelConfig(name="frozen"),
            graph=RingGraphConfig(n_agents=10),
            steps=10,
            metrics=MetricsConfig(record_every=10),
            opinion_initializer=UniformOpinionInitializerConfig(low=0.5, high=0.5),
        )

        result = SimulationRunner.from_config(config).run(seed=1)

        self.assertEqual(set(result.final_opinions), {0.5})

    def test_monte_carlo_aggregates_requested_runs(self) -> None:
        config = SimulationConfig(
            name="confidence",
            model=ATBCRModelConfig(epsilon=0.4, theta=0.9, mu=0.2),
            graph=ErdosRenyiGraphConfig(n_agents=25, er_probability=0.2),
            steps=100,
            metrics=MetricsConfig(record_every=100),
        )

        result = run_monte_carlo(config, MonteCarloConfig(runs=3, seed=10))

        self.assertEqual(result.runs, 3)
        self.assertEqual(len(result.simulations), 3)
        self.assertTrue(0 <= result.mean_extremized_share <= 1)


class PlotCliTests(unittest.TestCase):
    def test_metric_command_delegates_to_metric_plotter(self) -> None:
        with patch("atbcr_analysis.plot_cli.plot_metric_trajectories") as plot_metric:
            exit_code = plot_cli.main(
                [
                    "metric",
                    "runs/reference/trajectories.csv",
                    "--metric",
                    "mean_abs_opinion",
                    "--output",
                    "runs/reference/mean_abs.png",
                ],
            )

        self.assertEqual(exit_code, 0)
        plot_metric.assert_called_once_with(
            Path("runs/reference/trajectories.csv"),
            Path("runs/reference/mean_abs.png"),
            metric="mean_abs_opinion",
        )

    def test_metric_command_uses_default_metric(self) -> None:
        with patch("atbcr_analysis.plot_cli.plot_metric_trajectories") as plot_metric:
            exit_code = plot_cli.main(
                [
                    "metric",
                    "runs/reference/trajectories.csv",
                    "--output",
                    "runs/reference/max_abs.png",
                ],
            )

        self.assertEqual(exit_code, 0)
        plot_metric.assert_called_once_with(
            Path("runs/reference/trajectories.csv"),
            Path("runs/reference/max_abs.png"),
            metric="max_abs_opinion",
        )

    def test_distribution_command_delegates_to_distribution_plotter(self) -> None:
        with patch("atbcr_analysis.plot_cli.plot_final_distribution") as plot_distribution:
            exit_code = plot_cli.main(
                [
                    "distribution",
                    "runs/reference/snapshots.csv",
                    "--bins",
                    "20",
                    "--output",
                    "runs/reference/final_distribution.svg",
                ],
            )

        self.assertEqual(exit_code, 0)
        plot_distribution.assert_called_once_with(
            Path("runs/reference/snapshots.csv"),
            Path("runs/reference/final_distribution.svg"),
            bins=20,
        )

    def test_distribution_command_uses_default_bins(self) -> None:
        with patch("atbcr_analysis.plot_cli.plot_final_distribution") as plot_distribution:
            exit_code = plot_cli.main(
                [
                    "distribution",
                    "runs/reference/snapshots.csv",
                    "--output",
                    "runs/reference/final_distribution.png",
                ],
            )

        self.assertEqual(exit_code, 0)
        plot_distribution.assert_called_once_with(
            Path("runs/reference/snapshots.csv"),
            Path("runs/reference/final_distribution.png"),
            bins=50,
        )


if __name__ == "__main__":
    unittest.main()
