"""CLI for plotting persisted experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from atbcr_analysis.plots import (
    plot_final_distribution,
    plot_metric_trajectories,
    plot_opinion_trajectories,
)


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

    opinions = subparsers.add_parser("opinions")
    opinions.add_argument("snapshots_csv", type=Path)
    opinions.add_argument("--scenario")
    opinions.add_argument("--variant")
    opinions.add_argument("--normalizer")
    opinions.add_argument("--normalization-every")
    opinions.add_argument("--run-seed")
    opinions.add_argument("--max-agents", type=int)
    opinions.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "metric":
        plot_metric_trajectories(args.trajectories_csv, args.output, metric=args.metric)
    elif args.command == "distribution":
        plot_final_distribution(args.snapshots_csv, args.output, bins=args.bins)
    elif args.command == "opinions":
        plot_opinion_trajectories(
            args.snapshots_csv,
            args.output,
            scenario=args.scenario,
            variant=args.variant,
            normalizer=args.normalizer,
            normalization_every=args.normalization_every,
            run_seed=args.run_seed,
            max_agents=args.max_agents,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
