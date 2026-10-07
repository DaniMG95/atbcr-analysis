"""CLI for plotting persisted experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from atbcr_analysis.plots import plot_final_distribution, plot_metric_trajectories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atbcr-plot",
        description="Create plots from persisted ATBCR CSV outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    metric = subparsers.add_parser("metric")
    metric.add_argument("trajectories_csv", type=Path)
    metric.add_argument("--metric", default="max_abs_opinion")
    metric.add_argument("--output", type=Path, required=True)

    distribution = subparsers.add_parser("distribution")
    distribution.add_argument("snapshots_csv", type=Path)
    distribution.add_argument("--bins", type=int, default=50)
    distribution.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "metric":
        plot_metric_trajectories(args.trajectories_csv, args.output, metric=args.metric)
    elif args.command == "distribution":
        plot_final_distribution(args.snapshots_csv, args.output, bins=args.bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
