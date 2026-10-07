from __future__ import annotations

import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atbcr_analysis import cli
from atbcr_analysis import plot_cli
from atbcr_analysis import plots
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


def _fake_experiment_result() -> SimpleNamespace:
    return SimpleNamespace(
        scenario_name="repulsivo",
        variant_name="baseline_01",
        normalizer_name="none",
        normalization_every=None,
        mean_confidence_frequency=0.0,
        mean_inaction_frequency=0.0,
        mean_repulsion_frequency=0.0,
        mean_max_abs_opinion=0.0,
    )


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

    def test_monte_carlo_reports_progress_after_each_run(self) -> None:
        config = SimulationConfig(
            name="progress",
            model=ATBCRModelConfig(epsilon=0.4, theta=0.9, mu=0.2),
            graph=RingGraphConfig(n_agents=10),
            steps=5,
            metrics=MetricsConfig(record_every=5, store_opinion_snapshots=False),
        )
        progress_calls: list[tuple[int, int]] = []

        run_monte_carlo(
            config,
            MonteCarloConfig(runs=3, seed=10),
            progress_callback=lambda completed, total: progress_calls.append(
                (completed, total),
            ),
        )

        self.assertEqual(progress_calls, [(1, 3), (2, 3), (3, 3)])


class SimulationCliConfigTests(unittest.TestCase):
    def test_yaml_config_runs_single_experiment_and_allows_cli_overrides(self) -> None:
        config_path = Path("runs/test-cli-config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "\n".join(
                [
                    "scenarios:",
                    "  - repulsivo:0.2:0.7:0.1",
                    "variants:",
                    "  - baseline_01",
                    "steps: 10",
                    "runs: 1",
                    "workers: max",
                    "agents: 5",
                    "record_every: 10",
                    "no_snapshots: true",
                    "output_dir: runs/yaml-single",
                ],
            ),
            encoding="utf-8",
        )

        with (
            patch("atbcr_analysis.cli.run_monte_carlo", return_value=_fake_experiment_result())
            as run_experiment,
            patch("atbcr_analysis.cli.write_experiment_outputs") as write_outputs,
        ):
            exit_code = cli.main(["--config", str(config_path), "--seed", "99"])

        self.assertEqual(exit_code, 0)
        _, monte_carlo = run_experiment.call_args.args
        self.assertEqual(monte_carlo.seed, 99)
        self.assertEqual(monte_carlo.runs, 1)
        self.assertEqual(monte_carlo.workers, os.cpu_count() or 1)
        write_outputs.assert_called_once()
        self.assertEqual(write_outputs.call_args.args[0], Path("runs/yaml-single"))

    def test_yaml_config_runs_multiple_experiments_with_defaults(self) -> None:
        config_path = Path("runs/test-cli-config-multiple.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "\n".join(
                [
                    "steps: 10",
                    "agents: 5",
                    "defaults:",
                    "  variants: [baseline_01]",
                    "  runs: 1",
                    "  record_every: 10",
                    "  no_snapshots: true",
                    "experiments:",
                    "  - scenarios: [repulsivo:0.2:0.7:0.1]",
                    "    output_dir: runs/yaml-repulsivo",
                    "  - scenarios: [confianza:0.4:0.9:0.1]",
                    "    output_dir: runs/yaml-confianza",
                ],
            ),
            encoding="utf-8",
        )

        with (
            patch("atbcr_analysis.cli.run_monte_carlo", return_value=_fake_experiment_result()),
            patch("atbcr_analysis.cli.write_experiment_outputs") as write_outputs,
        ):
            exit_code = cli.main(["--config", str(config_path)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [call.args[0] for call in write_outputs.call_args_list],
            [Path("runs/yaml-repulsivo"), Path("runs/yaml-confianza")],
        )


class PlotCliTests(unittest.TestCase):
    def test_opinion_plot_title_uses_effective_model_parameters_and_range(self) -> None:
        config_path = Path("runs/title-test/config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "\n".join(
                [
                    "{",
                    '  "scenarios": [',
                    '    {"name": "repulsivo", "model": {"name": "atbcr",',
                    '     "epsilon": 0.2, "theta": 0.7, "mu": 0.1}}',
                    "  ],",
                    '  "variants": [',
                    '    {"name": "unbounded", "domain": "unbounded",',
                    '     "threshold_scale": 2.0,',
                    '     "initializer": {"kind": "uniform", "low": -1.0, "high": 1.0}}',
                    "  ]",
                    "}",
                ],
            ),
            encoding="utf-8",
        )

        title = plots._build_opinion_plot_title(
            config_path.with_name("snapshots.csv"),
            plots.SnapshotSeries("repulsivo", "unbounded", "max_abs", "100", "7"),
        )

        self.assertEqual(
            title,
            "epsilon=0.4, theta=1.4, mu=0.1; "
            "rango [-1.0, 1.0]; max_abs every 100; seed 7",
        )

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

    def test_opinions_command_delegates_to_opinion_plotter(self) -> None:
        with patch("atbcr_analysis.plot_cli.plot_opinion_trajectories") as plot_opinions:
            exit_code = plot_cli.main(
                [
                    "opinions",
                    "runs/reference/snapshots.csv",
                    "--scenario",
                    "repulsivo",
                    "--variant",
                    "unbounded",
                    "--normalizer",
                    "max_abs",
                    "--normalization-every",
                    "100",
                    "--run-seed",
                    "7",
                    "--max-agents",
                    "25",
                    "--output",
                    "runs/reference/opinions.png",
                ],
            )

        self.assertEqual(exit_code, 0)
        plot_opinions.assert_called_once_with(
            Path("runs/reference/snapshots.csv"),
            Path("runs/reference/opinions.png"),
            scenario="repulsivo",
            variant="unbounded",
            normalizer="max_abs",
            normalization_every="100",
            run_seed="7",
            max_agents=25,
        )


if __name__ == "__main__":
    unittest.main()
