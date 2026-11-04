from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atbcr_analysis.plots import plot_cli
from atbcr_analysis.plots.snapshot_metrics import (
    calculate_snapshot_opinion_metrics,
    write_snapshot_opinion_metrics_csv,
)


class SnapshotOpinionMetricsTests(unittest.TestCase):
    def test_calculates_metrics_per_step_with_effective_epsilon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshots_path = root / "snapshots.csv"
            snapshots_path.write_text(
                "scenario,variant,normalizer,normalization_every,run_seed,"
                "graph_seed,opinion_seed,dynamics_seed,step,agent,opinion\n"
                "e0p2_t0p7,unbounded,none,,7,107,207,307,0,0,-1.0\n"
                "e0p2_t0p7,unbounded,none,,7,107,207,307,0,1,-0.2\n"
                "e0p2_t0p7,unbounded,none,,7,107,207,307,0,2,0.2\n"
                "e0p2_t0p7,unbounded,none,,7,107,207,307,0,3,1.0\n"
                "e0p2_t0p7,unbounded,none,,7,107,207,307,10,0,-2.0\n"
                "e0p2_t0p7,unbounded,none,,7,107,207,307,10,1,-0.1\n"
                "e0p2_t0p7,unbounded,none,,7,107,207,307,10,2,0.1\n"
                "e0p2_t0p7,unbounded,none,,7,107,207,307,10,3,2.0\n"
                "e0p2_t0p7,unbounded,none,,8,108,208,308,0,0,0.0\n",
                encoding="utf-8",
            )
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "variants": [
                            {
                                "name": "unbounded",
                                "threshold_scale": 2.0,
                            },
                        ],
                    },
                ),
                encoding="utf-8",
            )

            metrics = calculate_snapshot_opinion_metrics(
                snapshots_path,
                epsilon=0.2,
                theta=0.7,
                variant="unbounded",
                normalizer="none",
                run_seed="7",
            )

        self.assertEqual([item.step for item in metrics], [0, 10])
        self.assertEqual([item.agent_count for item in metrics], [4, 4])
        self.assertEqual([item.proportion_abs_lt_1 for item in metrics], [0.5, 0.5])
        self.assertEqual([item.closest_to_zero_agent for item in metrics], [1, 1])
        self.assertEqual([item.min_abs_opinion for item in metrics], [0.2, 0.1])
        self.assertEqual(
            [
                (item.closest_pair_agent_i, item.closest_pair_agent_j)
                for item in metrics
            ],
            [(1, 2), (1, 2)],
        )
        self.assertAlmostEqual(metrics[0].min_pairwise_distance or 0.0, 0.4)
        self.assertAlmostEqual(metrics[1].min_pairwise_distance or 0.0, 0.2)
        self.assertEqual([item.confidence_epsilon for item in metrics], [0.4, 0.4])
        self.assertEqual(
            [item.confidence_pair_exists for item in metrics],
            [False, True],
        )

    def test_writes_compact_metrics_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshots_path = root / "snapshots.csv"
            output_path = root / "metrics.csv"
            snapshots_path.write_text(
                "scenario,variant,normalizer,normalization_every,run_seed,"
                "step,agent,opinion\n"
                "e0p3_t0p8,unbounded,none,,7,0,0,0.25\n",
                encoding="utf-8",
            )
            metrics = calculate_snapshot_opinion_metrics(
                snapshots_path,
                epsilon=0.3,
                theta=0.8,
                variant="unbounded",
                normalizer="none",
                run_seed="7",
            )

            write_snapshot_opinion_metrics_csv(output_path, metrics)
            with output_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["proportion_abs_lt_1"], "1.0")
        self.assertEqual(rows[0]["closest_to_zero_agent"], "0")
        self.assertEqual(rows[0]["min_abs_opinion"], "0.25")
        self.assertEqual(rows[0]["closest_pair_agent_i"], "")
        self.assertEqual(rows[0]["closest_pair_agent_j"], "")
        self.assertEqual(rows[0]["min_pairwise_distance"], "")
        self.assertEqual(rows[0]["confidence_pair_exists"], "False")

    def test_snapshot_metrics_command_filters_and_writes_output(self) -> None:
        derived_metrics = [object()]
        with (
            patch(
                "atbcr_analysis.plots.plot_cli.calculate_snapshot_opinion_metrics",
                return_value=derived_metrics,
            ) as calculate,
            patch(
                "atbcr_analysis.plots.plot_cli.write_snapshot_opinion_metrics_csv",
            ) as write,
        ):
            exit_code = plot_cli.main(
                [
                    "opinion-metrics",
                    "runs/experiment_0.3/final_opinions.csv",
                    "--epsilon",
                    "0.2",
                    "--theta",
                    "0.7",
                    "--variant",
                    "unbounded",
                    "--normalizer",
                    "none",
                    "--run-seed",
                    "7",
                    "--output",
                    "runs/experiment_0.3/snapshot_metrics.csv",
                ],
            )

        self.assertEqual(exit_code, 0)
        calculate.assert_called_once_with(
            Path("runs/experiment_0.3/final_opinions.csv"),
            epsilon=0.2,
            theta=0.7,
            variant="unbounded",
            normalizer="none",
            normalization_every=None,
            run_seed="7",
        )
        write.assert_called_once_with(
            Path("runs/experiment_0.3/snapshot_metrics.csv"),
            derived_metrics,
        )


if __name__ == "__main__":
    unittest.main()
