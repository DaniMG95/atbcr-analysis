from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atbcr_analysis.plots import (
    aggregate_trajectories,
    fit_replica_trajectories,
    plot_comparative_trajectories,
    read_comparative_trajectories,
)
from atbcr_analysis.plots import plot_cli


FIELDNAMES = [
    "scenario",
    "variant",
    "normalizer",
    "normalization_every",
    "run_seed",
    "step",
    "metric_value",
]


def _write_trajectories(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    *,
    scenario: str = "scenario-a",
    variant: str = "unbounded",
    normalizer: str = "none",
    normalization_every: str = "",
    run_seed: str = "1",
    step: int = 0,
    value: float = 1.0,
) -> dict[str, str | int | float]:
    return {
        "scenario": scenario,
        "variant": variant,
        "normalizer": normalizer,
        "normalization_every": normalization_every,
        "run_seed": run_seed,
        "step": step,
        "metric_value": value,
    }


class ComparativePlotsTests(unittest.TestCase):
    def test_transform_is_applied_per_replica_before_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectories.csv"
            _write_trajectories(
                source,
                [
                    _row(run_seed="1", value=1.0),
                    _row(run_seed="2", value=9.0),
                ],
            )

            trajectories = read_comparative_trajectories(
                [source],
                metric="metric_value",
                transform="log",
            )
            points = next(
                iter(aggregate_trajectories(trajectories, aggregate="mean").values()),
            )

        self.assertAlmostEqual(points[0].value, math.log(3.0))
        self.assertNotAlmostEqual(points[0].value, math.log(5.0))

    def test_filters_keep_sources_and_configurations_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            rows = [
                _row(scenario="keep", variant="unbounded", value=2.0),
                _row(scenario="drop", variant="bounded_m11", value=3.0),
            ]
            _write_trajectories(first, rows)
            _write_trajectories(second, rows)

            trajectories = read_comparative_trajectories(
                [first, second],
                metric="metric_value",
                variants=["unbounded"],
                normalizers=["none"],
                normalization_every=["none"],
            )

        self.assertEqual(len(trajectories), 2)
        self.assertEqual({series.source for series in trajectories}, {str(first), str(second)})
        self.assertEqual({series.scenario for series in trajectories}, {"keep"})

    def test_median_aggregation_uses_replica_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectories.csv"
            _write_trajectories(
                source,
                [
                    _row(run_seed="1", value=1.0),
                    _row(run_seed="2", value=100.0),
                    _row(run_seed="3", value=3.0),
                ],
            )
            trajectories = read_comparative_trajectories([source], metric="metric_value")
            point = next(
                iter(aggregate_trajectories(trajectories, aggregate="median").values()),
            )[0]

        self.assertEqual(point.value, 3.0)

    def test_epsilon_and_theta_filter_grid_scenario_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectories.csv"
            _write_trajectories(
                source,
                [
                    _row(scenario="e0p2_t0p7", value=2.0),
                    _row(scenario="e0p3_t0p8", value=3.0),
                ],
            )

            trajectories = read_comparative_trajectories(
                [source],
                metric="metric_value",
                parameter_pairs=[(0.2, 0.7)],
            )

        series = next(iter(trajectories))
        self.assertEqual((series.scenario, series.epsilon, series.theta), ("e0p2_t0p7", 0.2, 0.7))

    def test_parameter_filters_read_named_scenarios_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trajectories.csv"
            _write_trajectories(source, [_row(scenario="repulsivo", value=2.0)])
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "scenarios": [
                            {
                                "name": "repulsivo",
                                "model": {"epsilon": 0.25, "theta": 0.75},
                            },
                        ],
                    },
                ),
                encoding="utf-8",
            )

            trajectories = read_comparative_trajectories(
                [source],
                metric="metric_value",
                parameter_pairs=[(0.25, 0.75)],
            )

        series = next(iter(trajectories))
        self.assertEqual((series.epsilon, series.theta), (0.25, 0.75))

    def test_parameter_filter_reports_unresolvable_named_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectories.csv"
            _write_trajectories(source, [_row(scenario="repulsivo", value=2.0)])

            with self.assertRaisesRegex(ValueError, "cannot resolve epsilon/theta"):
                read_comparative_trajectories(
                    [source],
                    metric="metric_value",
                    parameter_pairs=[(0.2, 0.7)],
                )

    def test_ci_uses_replica_values_at_each_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectories.csv"
            _write_trajectories(
                source,
                [
                    _row(run_seed="1", value=1.0),
                    _row(run_seed="2", value=3.0),
                ],
            )
            trajectories = read_comparative_trajectories([source], metric="metric_value")
            point = next(
                iter(
                    aggregate_trajectories(
                        trajectories,
                        aggregate="mean",
                        include_ci=True,
                    ).values(),
                ),
            )[0]

        self.assertEqual(point.replicas, 2)
        self.assertAlmostEqual(point.value, 2.0)
        self.assertAlmostEqual(point.ci95_low or 0.0, 0.04)
        self.assertAlmostEqual(point.ci95_high or 0.0, 3.96)

    def test_log_transform_rejects_non_positive_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectories.csv"
            _write_trajectories(source, [_row(value=0.0)])

            with self.assertRaisesRegex(ValueError, "requires positive"):
                read_comparative_trajectories(
                    [source],
                    metric="metric_value",
                    transform="log10",
                )

    def test_linear_fit_is_per_replica_and_respects_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectories.csv"
            rows = []
            for run_seed, slope in (("1", 2.0), ("2", 3.0)):
                rows.extend(
                    _row(run_seed=run_seed, step=step, value=10 + slope * step)
                    for step in (0, 1, 2, 3)
                )
            _write_trajectories(source, rows)
            trajectories = read_comparative_trajectories([source], metric="metric_value")

            fits = fit_replica_trajectories(trajectories, fit_start=1, fit_end=3)

        self.assertEqual([fit.run_seed for fit in fits], ["1", "2"])
        self.assertEqual([fit.n_points for fit in fits], [3, 3])
        self.assertEqual([fit.fit_start for fit in fits], [1, 1])
        self.assertEqual([fit.fit_end for fit in fits], [3, 3])
        self.assertAlmostEqual(fits[0].slope, 2.0)
        self.assertAlmostEqual(fits[1].slope, 3.0)
        self.assertAlmostEqual(fits[0].r_squared, 1.0)

    def test_plot_writes_publication_graph_and_fit_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectories.csv"
            output = Path(directory) / "comparison.svg"
            _write_trajectories(
                source,
                [
                    _row(run_seed=seed, step=step, value=1 + int(seed) * step)
                    for seed in ("1", "2")
                    for step in (0, 1, 2)
                ],
            )

            plot_comparative_trajectories(
                [source],
                output,
                metric="metric_value",
                include_ci=True,
                fit=True,
            )
            fit_output = output.with_name("comparison_fits.csv")

            self.assertGreater(output.stat().st_size, 0)
            self.assertGreater(fit_output.stat().st_size, 0)
            with fit_output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["record_type"] for row in rows],
                ["replica", "replica", "summary"],
            )

    def test_compare_cli_delegates_all_options(self) -> None:
        with patch(
            "atbcr_analysis.plots.plot_cli.plot_comparative_trajectories",
        ) as plot_compare:
            exit_code = plot_cli.main(
                [
                    "compare",
                    "--input",
                    "a/trajectories.csv",
                    "b/trajectories.csv",
                    "--pair",
                    "0.2",
                    "0.7",
                    "--pair",
                    "0.3",
                    "0.8",
                    "--variants",
                    "unbounded",
                    "--normalizers",
                    "none",
                    "--normalization-every",
                    "none",
                    "--metric",
                    "max_abs_opinion",
                    "--transform",
                    "log",
                    "--aggregate",
                    "median",
                    "--ci",
                    "--fit",
                    "--fit-start",
                    "100",
                    "--fit-end",
                    "500",
                    "--fit-output",
                    "fits.csv",
                    "--output",
                    "comparison.pdf",
                ],
            )

        self.assertEqual(exit_code, 0)
        plot_compare.assert_called_once_with(
            [Path("a/trajectories.csv"), Path("b/trajectories.csv")],
            Path("comparison.pdf"),
            metric="max_abs_opinion",
            parameter_pairs=[(0.2, 0.7), (0.3, 0.8)],
            variants=["unbounded"],
            normalizers=["none"],
            normalization_every=["none"],
            transform="log",
            aggregate="median",
            include_ci=True,
            fit=True,
            fit_start=100,
            fit_end=500,
            fit_output=Path("fits.csv"),
        )


if __name__ == "__main__":
    unittest.main()
