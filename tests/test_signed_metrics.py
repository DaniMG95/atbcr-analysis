from __future__ import annotations

import contextlib
import csv
import io
import math
import tempfile
import unittest
from pathlib import Path

from atbcr_analysis.config import (
    ATBCRShiftModelConfig,
    CompleteGraphConfig,
    MetricsConfig,
    MonteCarloConfig,
    SimulationConfig,
    UniformOpinionInitializerConfig,
)
from atbcr_analysis.experiments import run_monte_carlo
from atbcr_analysis.metrics import population_std
from atbcr_analysis.persistence import write_experiment_outputs
from atbcr_analysis.plots import (
    calculate_signed_snapshot_metrics,
    plot_cli,
    signed_opinion_metrics,
    summarize_signed_metrics,
    write_signed_snapshot_metrics_csv,
)

SNAPSHOT_FIELDNAMES = [
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
]


def _snapshot_rows(
    scenario: str,
    run_seed: int,
    step: int,
    opinions: list[float],
    *,
    opinion_seed: int | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "scenario": scenario,
            "variant": "bounded_m11",
            "normalizer": "none",
            "normalization_every": "",
            "run_seed": run_seed,
            "graph_seed": run_seed,
            "opinion_seed": run_seed if opinion_seed is None else opinion_seed,
            "dynamics_seed": run_seed,
            "step": step,
            "agent": agent,
            "opinion": opinion,
        }
        for agent, opinion in enumerate(opinions)
    ]


def _write_snapshots(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SignedMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_known_snapshot_values(self) -> None:
        opinions = [-1.0, -0.9, -0.1, 0.0, 0.2, 0.85, 1.0]
        metrics = signed_opinion_metrics(opinions)

        self.assertEqual(metrics.agent_count, 7)
        self.assertAlmostEqual(metrics.mean_opinion, 0.05 / 7)
        self.assertAlmostEqual(metrics.median_opinion, 0.0)
        # Linear interpolation at position (n - 1) * q on the sorted sample.
        self.assertAlmostEqual(metrics.q10_opinion, -0.94)
        self.assertAlmostEqual(metrics.q25_opinion, -0.5)
        self.assertAlmostEqual(metrics.q75_opinion, 0.525)
        self.assertAlmostEqual(metrics.q90_opinion, 0.91)
        self.assertAlmostEqual(metrics.iqr_opinion, 1.025)
        self.assertAlmostEqual(metrics.width_q90_q10, 1.85)
        self.assertAlmostEqual(metrics.std_opinion, population_std(opinions))
        self.assertAlmostEqual(metrics.extreme_pos_share, 2 / 7)
        self.assertAlmostEqual(metrics.extreme_neg_share, 2 / 7)
        self.assertAlmostEqual(metrics.at_pos_boundary_share, 1 / 7)
        self.assertAlmostEqual(metrics.at_neg_boundary_share, 1 / 7)
        self.assertEqual(metrics.central_count, 3)
        self.assertAlmostEqual(metrics.central_mean, 0.1 / 3)
        self.assertAlmostEqual(metrics.central_median, 0.0)
        central_mean = 0.1 / 3
        expected_std = math.sqrt(
            sum((value - central_mean) ** 2 for value in (-0.1, 0.0, 0.2)) / 3,
        )
        self.assertAlmostEqual(metrics.central_std, expected_std)

    def test_boundary_tolerance_and_cutoff_are_configurable(self) -> None:
        metrics = signed_opinion_metrics(
            [1 - 1e-6, -1 + 1e-6, 0.5],
            extreme_cutoff=0.5,
            boundary_tol=1e-5,
        )
        self.assertAlmostEqual(metrics.at_pos_boundary_share, 1 / 3)
        self.assertAlmostEqual(metrics.at_neg_boundary_share, 1 / 3)
        self.assertAlmostEqual(metrics.extreme_pos_share, 2 / 3)
        self.assertEqual(metrics.central_count, 0)

    def test_matches_persisted_shift_trajectories(self) -> None:
        config = SimulationConfig(
            name="shift_small",
            model=ATBCRShiftModelConfig(
                # bounded_m11 thresholds after threshold_scale=2.0.
                epsilon=0.6,
                theta=1.6,
                mu=0.1,
                p=0.9,
                shift_lambda=0.01,
            ),
            graph=CompleteGraphConfig(n_agents=50),
            variant="bounded_m11",
            domain="bounded_m11",
            steps=2000,
            metrics=MetricsConfig(record_every=500, store_opinion_snapshots=True),
            opinion_initializer=UniformOpinionInitializerConfig(low=-1.0, high=1.0),
        )
        result = run_monte_carlo(config, MonteCarloConfig(runs=2, seed=11))
        write_experiment_outputs(self.dir, [result], {"test": True})

        output = self.dir / "signed.csv"
        written = write_signed_snapshot_metrics_csv(
            output,
            calculate_signed_snapshot_metrics(
                self.dir / "snapshots.csv",
                trajectories_csv=self.dir / "trajectories.csv",
            ),
        )
        signed_rows = _read_csv(output)
        trajectory_rows = {
            (row["run_seed"], row["step"]): row
            for row in _read_csv(self.dir / "trajectories.csv")
        }

        self.assertEqual(written, len(trajectory_rows))
        self.assertEqual(len(signed_rows), len(trajectory_rows))
        for row in signed_rows:
            trajectory = trajectory_rows[(row["run_seed"], row["step"])]
            self.assertAlmostEqual(
                float(row["std_opinion"]),
                float(trajectory["std_opinion"]),
                delta=1e-12,
            )
            self.assertAlmostEqual(
                float(row["extreme_pos_share"]) + float(row["extreme_neg_share"]),
                float(trajectory["extremized_share"]),
                delta=1e-12,
            )
            self.assertEqual(row["atbcr_steps"], trajectory["atbcr_steps"])
            self.assertEqual(row["shift_steps"], trajectory["shift_steps"])

    def test_reflection_flips_location_and_swaps_extremes(self) -> None:
        opinions = [-0.95, -0.3, 0.1, 0.4, 0.82, 0.9, 1.0]
        original = signed_opinion_metrics(opinions)
        reflected = signed_opinion_metrics([-value for value in opinions])

        self.assertAlmostEqual(reflected.mean_opinion, -original.mean_opinion)
        self.assertAlmostEqual(reflected.median_opinion, -original.median_opinion)
        self.assertAlmostEqual(reflected.extreme_pos_share, original.extreme_neg_share)
        self.assertAlmostEqual(reflected.extreme_neg_share, original.extreme_pos_share)
        self.assertAlmostEqual(reflected.at_pos_boundary_share, original.at_neg_boundary_share)
        self.assertNotAlmostEqual(original.mean_opinion, 0.0)

    def test_reappearing_block_raises(self) -> None:
        path = self.dir / "snapshots.csv"
        _write_snapshots(
            path,
            [
                *_snapshot_rows("a", 1, 0, [0.1, 0.2]),
                *_snapshot_rows("a", 1, 500, [0.1, 0.2]),
                *_snapshot_rows("a", 1, 0, [0.3]),
            ],
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            list(calculate_signed_snapshot_metrics(path))

    def test_scenario_filter_only_processes_selected_blocks(self) -> None:
        path = self.dir / "snapshots.csv"
        _write_snapshots(
            path,
            [
                *_snapshot_rows("keep", 1, 0, [0.5, 0.7]),
                *_snapshot_rows("skip", 1, 0, [0.1]),
                *_snapshot_rows("keep_too", 1, 0, [-0.5]),
            ],
        )
        records = list(calculate_signed_snapshot_metrics(path, scenarios=["keep"]))
        self.assertEqual([record.scenario for record in records], ["keep"])
        self.assertAlmostEqual(records[0].metrics.mean_opinion, 0.6)

    def test_baseline_deltas_are_paired_by_run_seed(self) -> None:
        snapshots = self.dir / "snapshots.csv"
        case_means = {1: 0.5, 2: 0.2, 3: -0.1}
        control_means = {1: 0.1, 2: 0.0, 3: -0.4}
        rows: list[dict[str, object]] = []
        for scenario, means in (("case", case_means), ("control", control_means)):
            for seed, value in means.items():
                rows += _snapshot_rows(scenario, seed, 0, [value, value])
        _write_snapshots(snapshots, rows)
        signed = self.dir / "signed.csv"
        write_signed_snapshot_metrics_csv(signed, calculate_signed_snapshot_metrics(snapshots))

        summary = summarize_signed_metrics(signed, baselines=[("case", "control")])
        case_row = next(row for row in summary.rows if row["scenario"] == "case")
        expected = sum(case_means[s] - control_means[s] for s in case_means) / 3
        self.assertAlmostEqual(case_row["delta_mean_opinion_mean"], expected)
        self.assertEqual(case_row["delta_runs"], 3)
        self.assertEqual(case_row["baseline_scenario"], "control")
        control_row = next(row for row in summary.rows if row["scenario"] == "control")
        self.assertNotIn("delta_mean_opinion_mean", control_row)

        output = self.dir / "summary.csv"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            plot_cli.main(
                [
                    "signed-metrics-summary",
                    str(signed),
                    "--baseline",
                    "case=control",
                    "--output",
                    str(output),
                ],
            )
        written = {row["scenario"]: row for row in _read_csv(output)}
        self.assertAlmostEqual(float(written["case"]["delta_mean_opinion_mean"]), expected)
        self.assertEqual(written["control"]["delta_mean_opinion_mean"], "")
        self.assertEqual(stderr.getvalue(), "")

    def test_baseline_with_different_opinion_seeds_raises(self) -> None:
        snapshots = self.dir / "snapshots.csv"
        _write_snapshots(
            snapshots,
            [
                *_snapshot_rows("case", 1, 0, [0.5]),
                *_snapshot_rows("control", 1, 0, [0.1], opinion_seed=99),
            ],
        )
        signed = self.dir / "signed.csv"
        write_signed_snapshot_metrics_csv(signed, calculate_signed_snapshot_metrics(snapshots))
        with self.assertRaisesRegex(ValueError, "not paired"):
            summarize_signed_metrics(signed, baselines=[("case", "control")])

    def test_missing_pairs_are_dropped_and_reported(self) -> None:
        snapshots = self.dir / "snapshots.csv"
        _write_snapshots(
            snapshots,
            [
                *_snapshot_rows("case", 1, 0, [0.5]),
                *_snapshot_rows("case", 2, 0, [0.6]),
                *_snapshot_rows("control", 1, 0, [0.1]),
            ],
        )
        signed = self.dir / "signed.csv"
        write_signed_snapshot_metrics_csv(signed, calculate_signed_snapshot_metrics(snapshots))
        summary = summarize_signed_metrics(signed, baselines=[("case", "control")])
        self.assertEqual(len(summary.dropped_pairs), 1)
        self.assertEqual(summary.dropped_pairs[0].runs, 1)
        case_row = next(row for row in summary.rows if row["scenario"] == "case")
        self.assertEqual(case_row["delta_runs"], 1)
        self.assertAlmostEqual(case_row["delta_mean_opinion_mean"], 0.4)

    def test_empty_central_group_is_nan_and_excluded_from_summary(self) -> None:
        self.assertTrue(math.isnan(signed_opinion_metrics([-1.0, 0.9]).central_mean))

        snapshots = self.dir / "snapshots.csv"
        _write_snapshots(
            snapshots,
            [
                *_snapshot_rows("a", 1, 0, [-1.0, -0.9, 0.85, 1.0]),
                *_snapshot_rows("a", 2, 0, [-1.0, 0.3, 1.0]),
            ],
        )
        signed = self.dir / "signed.csv"
        plot_cli.main(["signed-metrics", str(snapshots), "--output", str(signed)])
        rows = _read_csv(signed)
        self.assertEqual(rows[0]["central_count"], "0")
        self.assertTrue(math.isnan(float(rows[0]["central_mean"])))
        self.assertEqual(rows[0]["atbcr_steps"], "")

        (row,) = summarize_signed_metrics(signed).rows
        self.assertEqual(row["runs"], 2)
        self.assertEqual(row["central_mean_n"], 1)
        self.assertAlmostEqual(row["central_mean_mean"], 0.3)
        self.assertTrue(math.isnan(row["central_mean_ci95_low"]))
        self.assertEqual(row["mean_opinion_n"], 2)


if __name__ == "__main__":
    unittest.main()
