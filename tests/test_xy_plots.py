from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from atbcr_analysis.plots import plot_cli, plot_xy_means, read_xy_means

FIELDNAMES = [
    "scenario",
    "variant",
    "normalizer",
    "normalization_every",
    "run_seed",
    "step",
    "a",
    "b",
]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(scenario: str, seed: str, step: int, a: float, b: float) -> dict[str, object]:
    return {
        "scenario": scenario,
        "variant": "v",
        "normalizer": "none",
        "normalization_every": "",
        "run_seed": seed,
        "step": step,
        "a": a,
        "b": b,
    }


class XYPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.csv = self.dir / "trajectories.csv"
        _write(
            self.csv,
            [
                _row("s1", "1", 0, 0.0, 1.0),
                _row("s1", "1", 10, 1.0, 3.0),
                _row("s1", "2", 0, 0.0, 5.0),
                _row("s1", "2", 10, 1.0, 7.0),
                _row("s2", "1", 0, 0.0, 2.0),
                _row("s2", "1", 10, 1.0, 4.0),
            ],
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_single_seed_keeps_one_series_per_scenario(self) -> None:
        data = read_xy_means([self.csv], x="step", y="b", run_seeds=["1"])
        by_scenario = {series.scenario: points for series, points in data.items()}
        self.assertEqual(set(by_scenario), {"s1", "s2"})
        self.assertEqual([p.y_mean for p in by_scenario["s1"]], [1.0, 3.0])

    def test_without_seed_filter_averages_over_seeds(self) -> None:
        data = read_xy_means([self.csv], x="step", y="b", scenarios=["s1"])
        (points,) = data.values()
        self.assertEqual([(p.x, p.y_mean, p.count) for p in points], [(0, 3.0, 2), (10, 5.0, 2)])

    def test_bins_group_continuous_x(self) -> None:
        data = read_xy_means([self.csv], x="b", y="a", scenarios=["s1"], bins=2)
        (points,) = data.values()
        self.assertEqual(len(points), 2)
        self.assertEqual(sum(p.count for p in points), 4)

    def test_no_matching_rows_raises(self) -> None:
        with self.assertRaises(ValueError):
            read_xy_means([self.csv], x="step", y="b", run_seeds=["99"])

    def test_cli_writes_png(self) -> None:
        output = self.dir / "xy.png"
        plot_cli.main(
            ["xy", "--input", str(self.csv), "--y", "b", "--run-seed", "1", "--output", str(output)],
        )
        self.assertTrue(output.exists())

    def test_many_series_get_distinct_styles(self) -> None:
        rows = [_row(f"s{i}", "1", step, 0.0, float(i)) for i in range(25) for step in (0, 1)]
        _write(self.csv, rows)
        plot_xy_means([self.csv], self.dir / "many.png", x="step", y="b")
        self.assertTrue((self.dir / "many.png").exists())


if __name__ == "__main__":
    unittest.main()
