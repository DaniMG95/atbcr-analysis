from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atbcr_analysis.plots import (
    SnapshotInterval,
    aggregate_snapshot_stability,
    plot_snapshot_stability,
    read_snapshot_stability,
    read_snapshot_stability_csv,
)
from atbcr_analysis.plots import plot_cli


FIELDNAMES = [
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


def _write_snapshots(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _run_rows(
    *,
    scenario: str = "e0p2_t0p7",
    frequency: str,
    run_seed: int,
    snapshots: list[tuple[int, list[float]]],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for step, opinions in snapshots:
        for agent, opinion in enumerate(opinions):
            rows.append(
                {
                    "scenario": scenario,
                    "variant": "unbounded",
                    "normalizer": "max_abs",
                    "normalization_every": frequency,
                    "run_seed": run_seed,
                    "graph_seed": run_seed + 100,
                    "opinion_seed": run_seed + 200,
                    "dynamics_seed": run_seed + 300,
                    "step": step,
                    "agent": agent,
                    "opinion": opinion,
                },
            )
    return rows


class SnapshotStabilityTests(unittest.TestCase):
    def test_calculates_per_run_deltas_and_averages_exact_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "snapshots.csv"
            rows = [
                *_run_rows(
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [0.0, 1.0]), (5, [0.3, 0.9]), (20, [0.4, 0.4])],
                ),
                *_run_rows(
                    frequency="10",
                    run_seed=2,
                    snapshots=[
                        (0, [10.0, 20.0]),
                        (5, [10.1, 20.4]),
                        (20, [9.8, 20.2]),
                    ],
                ),
            ]
            _write_snapshots(path, rows)

            stability = read_snapshot_stability([path])
            aggregates = aggregate_snapshot_stability(stability)

        self.assertEqual(len(stability), 1)
        runs = next(iter(stability.values()))
        self.assertEqual(len(runs), 2)
        self.assertAlmostEqual(
            next(iter(runs.values()))[SnapshotInterval(0, 5)],
            0.3,
        )
        points = next(iter(aggregates.values()))
        self.assertEqual([(point.previous_step, point.step) for point in points], [(0, 5), (5, 20)])
        self.assertEqual([point.replicas for point in points], [2, 2])
        self.assertAlmostEqual(points[0].value, 0.35)
        self.assertAlmostEqual(points[1].value, 0.4)

    def test_filters_and_keeps_normalization_frequencies_as_separate_series(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "snapshots.csv"
            rows = [
                *_run_rows(
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [0.0]), (10, [0.2])],
                ),
                *_run_rows(
                    frequency="100",
                    run_seed=1,
                    snapshots=[(0, [0.0]), (10, [0.8])],
                ),
                *_run_rows(
                    scenario="other",
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [0.0]), (10, [9.0])],
                ),
            ]
            _write_snapshots(path, rows)

            stability = read_snapshot_stability(
                [path],
                scenarios=["e0p2_t0p7"],
                variants=["unbounded"],
                normalizers=["max_abs"],
                normalization_every=["10", "100"],
            )

        self.assertEqual(
            {series.normalization_every for series in stability},
            {"10", "100"},
        )
        self.assertEqual({series.scenario for series in stability}, {"e0p2_t0p7"})

    def test_selects_one_run_per_configuration_and_reports_chunk_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "snapshots.csv"
            rows = [
                *_run_rows(
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [0.0]), (10, [0.2])],
                ),
                *_run_rows(
                    frequency="10",
                    run_seed=2,
                    snapshots=[(0, [0.0]), (10, [9.0])],
                ),
                *_run_rows(
                    frequency="100",
                    run_seed=1,
                    snapshots=[(0, [0.0]), (10, [0.8])],
                ),
            ]
            _write_snapshots(path, rows)
            progress: list[tuple[int, int]] = []

            stability = read_snapshot_stability(
                [path],
                scenarios=["e0p2_t0p7"],
                variants=["unbounded"],
                normalizers=["max_abs"],
                normalization_every=["10", "100"],
                run_seed="1",
                chunk_size=1,
                progress=lambda _path, processed, total: progress.append((processed, total)),
            )

        self.assertEqual({len(runs) for runs in stability.values()}, {1})
        self.assertEqual(
            sorted(
                next(iter(runs.values()))[SnapshotInterval(0, 10)]
                for runs in stability.values()
            ),
            [0.2, 0.8],
        )
        self.assertGreater(len(progress), 2)
        self.assertEqual(progress[0][0], 0)
        self.assertTrue(all(total == progress[0][1] for _, total in progress))

    def test_accepts_multiple_input_files_without_combining_equal_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.csv"
            second = root / "second.csv"
            _write_snapshots(
                first,
                _run_rows(
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [0.0]), (10, [0.2])],
                ),
            )
            _write_snapshots(
                second,
                _run_rows(
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [100.0]), (10, [101.0])],
                ),
            )

            stability = read_snapshot_stability([first, second])
            aggregates = aggregate_snapshot_stability(stability)

        self.assertEqual(len(stability), 2)
        self.assertEqual(
            sorted(points[0].value for points in aggregates.values()),
            [0.2, 1.0],
        )

    def test_rejects_different_agent_sets_between_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "snapshots.csv"
            _write_snapshots(
                path,
                _run_rows(
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [0.0, 1.0]), (10, [0.2])],
                ),
            )

            with self.assertRaisesRegex(ValueError, "agent sets differ"):
                read_snapshot_stability([path])

    def test_plot_explicitly_describes_consecutive_snapshot_intervals(self) -> None:
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshots = root / "snapshots.csv"
            output = root / "stability.png"
            _write_snapshots(
                snapshots,
                _run_rows(
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [0.0]), (10, [0.2])],
                ),
            )
            with patch("matplotlib.pyplot.close") as close_plot:
                plot_snapshot_stability([snapshots], output)

            figure = close_plot.call_args.args[0]
            try:
                axis = figure.axes[0]
                self.assertIn("snapshots consecutivos", axis.get_title())
                self.assertIn("snapshot final del intervalo", axis.get_xlabel())
                self.assertTrue(
                    any(
                        "no representa necesariamente un solo paso" in text.get_text()
                        for text in figure.texts
                    ),
                )
                self.assertGreater(output.stat().st_size, 0)
            finally:
                plt.close(figure)

    def test_compact_cache_is_reused_without_reading_snapshots_again(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshots = root / "snapshots.csv"
            cache = root / "stability.csv"
            first_output = root / "first.png"
            second_output = root / "second.png"
            _write_snapshots(
                snapshots,
                _run_rows(
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [0.0]), (10, [0.2])],
                ),
            )
            options = {
                "scenarios": ["e0p2_t0p7"],
                "variants": ["unbounded"],
                "normalizers": ["max_abs"],
                "normalization_every": ["10"],
                "run_seed": "1",
                "cache_path": cache,
            }
            plot_snapshot_stability([snapshots], first_output, **options)
            cached = read_snapshot_stability_csv(cache)
            statuses: list[str] = []

            with patch(
                "atbcr_analysis.plots.stability.read_snapshot_stability",
                side_effect=AssertionError("snapshots should not be read when cache is current"),
            ):
                plot_snapshot_stability(
                    [snapshots],
                    second_output,
                    status=statuses.append,
                    **options,
                )
            first_size = first_output.stat().st_size
            second_size = second_output.stat().st_size

        self.assertTrue(cached)
        self.assertGreater(first_size, 0)
        self.assertGreater(second_size, 0)
        self.assertTrue(any("Reutilizando cache" in status for status in statuses))

    def test_cache_is_rebuilt_when_selected_run_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshots = root / "snapshots.csv"
            cache = root / "stability.csv"
            rows = [
                *_run_rows(
                    frequency="10",
                    run_seed=1,
                    snapshots=[(0, [0.0]), (10, [0.2])],
                ),
                *_run_rows(
                    frequency="10",
                    run_seed=2,
                    snapshots=[(0, [0.0]), (10, [0.8])],
                ),
            ]
            _write_snapshots(snapshots, rows)
            common = {
                "scenarios": ["e0p2_t0p7"],
                "variants": ["unbounded"],
                "normalizers": ["max_abs"],
                "normalization_every": ["10"],
                "cache_path": cache,
            }
            plot_snapshot_stability(
                [snapshots],
                root / "run1.png",
                run_seed="1",
                **common,
            )
            statuses: list[str] = []
            plot_snapshot_stability(
                [snapshots],
                root / "run2.png",
                run_seed="2",
                status=statuses.append,
                **common,
            )
            cached = read_snapshot_stability_csv(cache)

        run_keys = [run_key for runs in cached.values() for run_key in runs]
        self.assertEqual({run_key[0] for run_key in run_keys}, {"2"})
        self.assertTrue(any("Generando cache" in status for status in statuses))

    def test_stability_command_delegates_to_plotter(self) -> None:
        with patch(
            "atbcr_analysis.plots.plot_cli.plot_snapshot_stability",
        ) as plot_stability:
            exit_code = plot_cli.main(
                [
                    "stability",
                    "--input",
                    "runs/a/snapshots.csv",
                    "runs/b/snapshots.csv",
                    "--scenarios",
                    "e0p2_t0p7",
                    "--variants",
                    "unbounded",
                    "--normalizers",
                    "max_abs",
                    "--normalization-every",
                    "10",
                    "100",
                    "--run-seed",
                    "7",
                    "--chunk-size",
                    "2000",
                    "--cache",
                    "runs/comparison/stability.csv",
                    "--refresh-cache",
                    "--no-progress",
                    "--output",
                    "runs/comparison/stability.svg",
                ],
            )

        self.assertEqual(exit_code, 0)
        positional, keyword = plot_stability.call_args
        self.assertEqual(
            positional,
            (
                [Path("runs/a/snapshots.csv"), Path("runs/b/snapshots.csv")],
                Path("runs/comparison/stability.svg"),
            ),
        )
        status = keyword.pop("status")
        self.assertTrue(callable(status))
        self.assertEqual(
            keyword,
            {
                "scenarios": ["e0p2_t0p7"],
                "variants": ["unbounded"],
                "normalizers": ["max_abs"],
                "normalization_every": ["10", "100"],
                "run_seed": "7",
                "chunk_size": 2000,
                "cache_path": Path("runs/comparison/stability.csv"),
                "refresh_cache": True,
                "progress": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
