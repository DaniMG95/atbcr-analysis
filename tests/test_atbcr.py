from __future__ import annotations

import csv
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from atbcr_analysis import analysis_cli
from atbcr_analysis import cli
from atbcr_analysis import plot_cli
from atbcr_analysis import plots
from atbcr_analysis.analysis import SeededFinalDistribution, paired_wasserstein_by_seed
from atbcr_analysis.config import (
    ATBCRModelConfig,
    BarabasiAlbertGraphConfig,
    CompleteGraphConfig,
    ErdosRenyiGraphConfig,
    MetricsConfig,
    ModelConfig,
    MonteCarloConfig,
    RingGraphConfig,
    SimulationConfig,
    UniformOpinionInitializerConfig,
)
from atbcr_analysis.experiments import run_monte_carlo
from atbcr_analysis.graphs import complete_graph_edges, generate_graph, graph_edges
from atbcr_analysis.metrics import cluster_count, extremized_share, percentile, summarize_values
from atbcr_analysis.models import register_model
from atbcr_analysis.normalizers import NormalizationConfig
from atbcr_analysis.persistence import write_experiment_outputs
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
    def test_complete_graph_edges_are_lazy_and_cover_each_pair_once(self) -> None:
        edges = complete_graph_edges(4)

        self.assertEqual(len(edges), 6)
        self.assertEqual(
            {edges[index] for index in range(len(edges))},
            {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)},
        )

    def test_barabasi_albert_generates_edges(self) -> None:
        import random

        graph = generate_graph(BarabasiAlbertGraphConfig(n_agents=20, ba_m=2), random.Random(1))

        self.assertEqual(len(graph), 20)
        self.assertGreater(len(graph_edges(graph)), 0)


class SimulationTests(unittest.TestCase):
    def test_atbcr_requires_epsilon_strictly_smaller_than_theta(self) -> None:
        with self.assertRaises(ValidationError):
            ATBCRModelConfig(epsilon=0.5, theta=0.5, mu=0.1)

    def test_extremized_share_uses_both_edges_for_unit_interval(self) -> None:
        opinions = [0.0, 0.1, 0.11, 0.89, 0.9, 1.0]

        share = extremized_share(opinions, threshold=123.0, domain="bounded_01")

        self.assertEqual(share, 4 / 6)

    def test_extremized_share_uses_absolute_value_for_centered_domains(self) -> None:
        opinions = [-0.81, -0.79, 0.00, 0.79, 0.80]

        share = extremized_share(opinions, threshold=123.0, domain="bounded_m11")

        self.assertEqual(share, 2 / 5)

    def test_extremized_share_uses_configurable_unbounded_cutoff(self) -> None:
        opinions = [-2.0, -0.9, 0.0, 1.1, 3.0]

        share = extremized_share(opinions, unbounded_extreme_cutoff=1.0, domain="unbounded")

        self.assertEqual(share, 3 / 5)

    def test_legacy_extremized_threshold_config_maps_to_unbounded_cutoff(self) -> None:
        metrics = MetricsConfig(extremized_threshold=2.0)

        self.assertEqual(metrics.unbounded_extreme_cutoff, 2.0)

    def test_cluster_count_documents_anchor_rule(self) -> None:
        opinions = [0.0, 0.0008, 0.0016]

        self.assertEqual(cluster_count(opinions, tolerance=0.001), 2)

    def test_percentile_interpolates_abs_opinion_statistics(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 90), 9.0)

    def test_baseline_01_and_bounded_m11_are_mathematically_equivalent(self) -> None:
        scenario = ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.1)
        metrics = MetricsConfig(record_every=10, cluster_tolerance=0.001)
        common = {
            "name": "equivalence",
            "graph": CompleteGraphConfig(n_agents=25),
            "steps": 50,
            "metrics": metrics,
        }
        baseline = SimulationConfig(
            **common,
            model=scenario,
            variant="baseline_01",
            domain="bounded_01",
            opinion_initializer=UniformOpinionInitializerConfig(low=0.0, high=1.0),
        )
        centered = SimulationConfig(
            **common,
            model=scenario.scale_thresholds(2.0),
            variant="bounded_m11",
            domain="bounded_m11",
            opinion_initializer=UniformOpinionInitializerConfig(low=-1.0, high=1.0),
        )

        baseline_result = simulate_atbcr(baseline, seed=123)
        centered_result = simulate_atbcr(centered, seed=123)

        self.assertEqual(baseline_result.graph_seed, centered_result.graph_seed)
        self.assertEqual(baseline_result.opinion_seed, centered_result.opinion_seed)
        self.assertEqual(baseline_result.dynamics_seed, centered_result.dynamics_seed)
        for baseline_snapshot, centered_snapshot in zip(
            baseline_result.trajectory,
            centered_result.trajectory,
            strict=True,
        ):
            self.assertIsNotNone(baseline_snapshot.opinions)
            self.assertIsNotNone(centered_snapshot.opinions)
            for x_opinion, y_opinion in zip(
                baseline_snapshot.opinions or (),
                centered_snapshot.opinions or (),
                strict=True,
            ):
                self.assertAlmostEqual(y_opinion, 2 * x_opinion - 1)
            self.assertAlmostEqual(
                centered_snapshot.metrics.confidence_frequency,
                baseline_snapshot.metrics.confidence_frequency,
            )
            self.assertAlmostEqual(
                centered_snapshot.metrics.inaction_frequency,
                baseline_snapshot.metrics.inaction_frequency,
            )
            self.assertAlmostEqual(
                centered_snapshot.metrics.repulsion_frequency,
                baseline_snapshot.metrics.repulsion_frequency,
            )
            self.assertEqual(
                centered_snapshot.metrics.cluster_count,
                baseline_snapshot.metrics.cluster_count,
            )
            self.assertAlmostEqual(
                centered_snapshot.metrics.extremized_share,
                baseline_snapshot.metrics.extremized_share,
            )
            self.assertEqual(centered_snapshot.metrics.effective_cluster_tolerance, 0.002)

    def test_window_frequencies_use_snapshot_interval(self) -> None:
        config = SimulationConfig(
            name="window",
            model=ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.1),
            graph=CompleteGraphConfig(n_agents=10),
            steps=12,
            metrics=MetricsConfig(record_every=5),
        )

        result = simulate_atbcr(config, seed=7)

        for snapshot in result.trajectory[1:]:
            metric = snapshot.metrics
            total = (
                metric.confidence_frequency_window
                + metric.inaction_frequency_window
                + metric.repulsion_frequency_window
            )
            self.assertAlmostEqual(total, 1.0)

    def test_normalization_events_capture_before_after_and_scale(self) -> None:
        config = SimulationConfig(
            name="normalization",
            model=ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.1),
            graph=CompleteGraphConfig(n_agents=10),
            variant="unbounded",
            domain="unbounded",
            steps=4,
            metrics=MetricsConfig(record_every=2),
            opinion_initializer=UniformOpinionInitializerConfig(low=-2.0, high=2.0),
            normalization=NormalizationConfig(kind="max_abs", every=2),
        )

        result = simulate_atbcr(config, seed=9)

        self.assertEqual([event.step for event in result.normalization_events], [2, 4])
        for event in result.normalization_events:
            self.assertEqual(event.normalizer, "max_abs")
            self.assertEqual(event.scale, event.max_abs_before)
            self.assertAlmostEqual(event.max_abs_after, 1.0)

    def test_active_normalizer_requires_positive_interval(self) -> None:
        with self.assertRaises(ValueError):
            NormalizationConfig(kind="max_abs")
        with self.assertRaises(ValueError):
            NormalizationConfig(kind="max_abs", every=0)
        with self.assertRaises(ValueError):
            NormalizationConfig(kind="signed_log_max_abs", every=-10)

    def test_complete_graph_simulation_does_not_materialize_edges(self) -> None:
        config = SimulationConfig(
            name="complete",
            model=ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.1),
            graph=CompleteGraphConfig(n_agents=20),
            steps=2,
            metrics=MetricsConfig(record_every=2),
        )

        with patch("atbcr_analysis.simulation.graph_edges") as materialize_edges:
            result = simulate_atbcr(config, seed=1)

        materialize_edges.assert_not_called()
        self.assertEqual(result.trajectory[-1].metrics.step, 2)

    def test_interaction_events_are_optional_and_skip_inaction(self) -> None:
        config = SimulationConfig(
            name="events",
            model=ATBCRModelConfig(epsilon=2.0, theta=3.0, mu=0.1),
            graph=CompleteGraphConfig(n_agents=8),
            steps=3,
            metrics=MetricsConfig(record_every=3, store_interaction_events=True),
        )

        result = simulate_atbcr(config, seed=2)

        self.assertEqual(len(result.interaction_events), 3)
        self.assertEqual({event.outcome for event in result.interaction_events}, {"confidence"})
        self.assertEqual(result.interaction_events[0].step, 1)

    def test_runner_accepts_explicit_phase_seeds(self) -> None:
        config = SimulationConfig(
            name="seeds",
            model=ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.1),
            graph=CompleteGraphConfig(n_agents=10),
            steps=5,
            metrics=MetricsConfig(record_every=5),
        )

        result = SimulationRunner.from_config(config).run(
            1,
            graph_seed=10,
            opinion_seed=20,
            dynamics_seed=30,
        )

        self.assertEqual(result.graph_seed, 10)
        self.assertEqual(result.opinion_seed, 20)
        self.assertEqual(result.dynamics_seed, 30)

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
        self.assertIn("p95_abs_opinion", result.metric_summaries)
        self.assertIn("confidence_frequency_window", result.metric_summaries)

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

    def test_paired_wasserstein_requires_matching_seed_sets_and_summarizes(self) -> None:
        result = paired_wasserstein_by_seed(
            {1: [0.0, 1.0], 2: [0.0, 2.0]},
            {1: [0.5, 1.5], 2: [1.0, 3.0]},
        )

        self.assertEqual(result.distances_by_seed, {1: 0.5, 2: 1.0})
        self.assertAlmostEqual(result.summary.mean, 0.75)
        self.assertAlmostEqual(result.summary.median, 0.75)
        with self.assertRaises(ValueError):
            paired_wasserstein_by_seed({1: [0.0]}, {2: [0.0]})
        with self.assertRaisesRegex(ValueError, "mismatched graph_seed"):
            paired_wasserstein_by_seed(
                {1: SeededFinalDistribution([0.0], graph_seed=1)},
                {1: SeededFinalDistribution([0.0], graph_seed=2)},
            )

    def test_metric_summary_marks_ci95_unavailable_for_single_run(self) -> None:
        summary = summarize_values([1.25])

        self.assertEqual(summary.mean, 1.25)
        self.assertEqual(summary.median, 1.25)
        self.assertNotEqual(summary.ci95_low, summary.ci95_low)
        self.assertNotEqual(summary.ci95_high, summary.ci95_high)

    def test_wasserstein_cli_writes_summary_and_by_seed_distances(self) -> None:
        snapshots_path = Path("runs/test-wasserstein/snapshots.csv")
        snapshots_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshots_path.open("w", newline="", encoding="utf-8") as handle:
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
            for variant, offset in [("baseline_01", 0.0), ("bounded_m11", 0.5)]:
                for run_seed in [7, 8]:
                    for agent, opinion in enumerate([0.0 + offset, 1.0 + offset]):
                        writer.writerow(
                            {
                                "scenario": "s",
                                "variant": variant,
                                "normalizer": "none",
                                "normalization_every": "",
                                "run_seed": run_seed,
                                "graph_seed": 100 + run_seed,
                                "opinion_seed": 200 + run_seed,
                                "dynamics_seed": 300 + run_seed,
                                "step": 10,
                                "agent": agent,
                                "opinion": opinion,
                            },
                        )

        output_path = snapshots_path.with_name("wasserstein.csv")
        exit_code = analysis_cli.main([str(snapshots_path), "--output", str(output_path)])

        self.assertEqual(exit_code, 0)
        with output_path.open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        with output_path.with_name("wasserstein_by_seed.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            by_seed_rows = list(csv.DictReader(handle))
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(len(by_seed_rows), 2)
        self.assertEqual(by_seed_rows[0]["wasserstein_distance"], "0.5")

    def test_persistence_includes_new_metrics_seeds_and_normalization_events(self) -> None:
        config = SimulationConfig(
            name="persist",
            model=ATBCRModelConfig(epsilon=0.2, theta=0.7, mu=0.1),
            graph=CompleteGraphConfig(n_agents=8),
            variant="unbounded",
            domain="unbounded",
            steps=2,
            metrics=MetricsConfig(record_every=1, store_opinion_snapshots=False),
            opinion_initializer=UniformOpinionInitializerConfig(low=-2.0, high=2.0),
            normalization=NormalizationConfig(kind="max_abs", every=1),
        )
        result = run_monte_carlo(config, MonteCarloConfig(runs=1, seed=10))
        output_dir = Path("runs/test-persistence")

        write_experiment_outputs(output_dir, [result], {"test": True})

        with (output_dir / "trajectories.csv").open(newline="", encoding="utf-8") as handle:
            trajectory_header = next(csv.reader(handle))
        with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
            summary_header = next(csv.reader(handle))
        with (output_dir / "normalization_events.csv").open(
            newline="",
            encoding="utf-8",
        ) as handle:
            event_rows = list(csv.DictReader(handle))
        self.assertIn("confidence_frequency_window", trajectory_header)
        self.assertIn("p95_abs_opinion", trajectory_header)
        self.assertIn("graph_seed", trajectory_header)
        self.assertIn("p95_abs_opinion_mean", summary_header)
        self.assertIn("confidence_frequency_window_mean", summary_header)
        self.assertIn("max_abs_opinion_ci95_high", summary_header)
        self.assertIn("nan", (output_dir / "summary.csv").read_text(encoding="utf-8"))
        self.assertEqual(len(event_rows), 2)
        self.assertEqual(event_rows[0]["normalizer"], "max_abs")


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

    def test_yaml_config_expands_scenario_grid(self) -> None:
        config_path = Path("runs/test-cli-scenario-grid.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "\n".join(
                [
                    "scenario_grid:",
                    "  epsilon: {start: 0, stop: 0.05, step: 0.05}",
                    "  theta: {start: 0, stop: 0.1, step: 0.05}",
                    "  epsilon_less_than_theta: true",
                    "  mu: 0.1",
                    "variants: [baseline_01]",
                    "steps: 10",
                    "runs: 1",
                    "agents: 5",
                    "record_every: 10",
                    "no_snapshots: true",
                    "output_dir: runs/yaml-grid",
                ],
            ),
            encoding="utf-8",
        )

        with (
            patch("atbcr_analysis.cli.run_monte_carlo", return_value=_fake_experiment_result())
            as run_experiment,
            patch("atbcr_analysis.cli.write_experiment_outputs"),
        ):
            exit_code = cli.main(["--config", str(config_path)])

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_experiment.call_count, 3)


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
