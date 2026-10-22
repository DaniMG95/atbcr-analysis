"""CLI for plotting persisted experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from atbcr_analysis.persistence import reduce_snapshots_csv
from atbcr_analysis.plots import (
    plot_comparative_trajectories,
    plot_extremized_share_heatmap,
    plot_extremized_share_trajectory,
    plot_final_distribution,
    plot_final_opinion_histogram,
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

    compare = subparsers.add_parser(
        "compare",
        help="compare transformed Monte Carlo metric trajectories",
    )
    compare.add_argument("--input", type=Path, nargs="+", required=True)
    _add_pair_argument(compare)
    compare.add_argument("--variants", nargs="+")
    compare.add_argument("--normalizers", nargs="+")
    compare.add_argument("--normalization-every", nargs="+")
    compare.add_argument("--metric", default="max_abs_opinion")
    compare.add_argument("--transform", choices=("none", "log", "log10"), default="none")
    compare.add_argument("--aggregate", choices=("mean", "median"), default="mean")
    compare.add_argument(
        "--ci",
        action="store_true",
        help="show normal-approximation 95 percent confidence intervals",
    )
    compare.add_argument("--fit", action="store_true", help="fit each replica independently")
    compare.add_argument("--fit-start", type=int)
    compare.add_argument("--fit-end", type=int)
    compare.add_argument("--fit-output", type=Path)
    compare.add_argument("--output", type=Path, required=True)

    distribution = subparsers.add_parser("distribution")
    distribution.add_argument("snapshots_csv", type=Path)
    distribution.add_argument("--bins", type=int, default=50)
    distribution.add_argument("--output", type=Path, required=True)

    final_opinions = subparsers.add_parser("final-opinions")
    final_opinions.add_argument("snapshots_csv", type=Path)
    final_opinions.add_argument("--output", type=Path)

    opinions = subparsers.add_parser("opinions")
    opinions.add_argument("snapshots_csv", type=Path)
    opinions.add_argument("--scenario")
    opinions.add_argument("--variant")
    opinions.add_argument("--normalizer")
    opinions.add_argument("--normalization-every")
    opinions.add_argument("--run-seed")
    opinions.add_argument("--max-agents", type=int)
    opinions.add_argument("--output", type=Path, required=True)

    heatmap = subparsers.add_parser("extremized-heatmap")
    heatmap.add_argument("summary_csv", type=Path)
    heatmap.add_argument("--variant")
    heatmap.add_argument("--normalizer")
    heatmap.add_argument("--normalization-every")
    heatmap.add_argument("--output", type=Path, required=True)

    parameter_distribution = subparsers.add_parser("parameter-distribution")
    parameter_distribution.add_argument("snapshots_csv", type=Path)
    _add_parameter_arguments(parameter_distribution, required=False)
    _add_pair_argument(parameter_distribution)
    parameter_distribution.add_argument("--bins", type=int, default=50)
    parameter_distribution.add_argument("--output", type=Path, required=True)

    extremized_trajectory = subparsers.add_parser("extremized-trajectory")
    extremized_trajectory.add_argument("trajectories_csv", type=Path)
    _add_parameter_arguments(extremized_trajectory, required=False)
    _add_pair_argument(extremized_trajectory)
    extremized_trajectory.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "metric":
        plot_metric_trajectories(args.trajectories_csv, args.output, metric=args.metric)
    elif args.command == "compare":
        if not args.fit and (args.fit_start is not None or args.fit_end is not None):
            parser.error("--fit-start and --fit-end require --fit")
        if args.fit_output is not None and not args.fit:
            parser.error("--fit-output requires --fit")
        plot_comparative_trajectories(
            args.input,
            args.output,
            metric=args.metric,
            parameter_pairs=[(pair[0], pair[1]) for pair in args.pair],
            variants=args.variants,
            normalizers=args.normalizers,
            normalization_every=args.normalization_every,
            transform=args.transform,
            aggregate=args.aggregate,
            include_ci=args.ci,
            fit=args.fit,
            fit_start=args.fit_start,
            fit_end=args.fit_end,
            fit_output=args.fit_output,
        )
    elif args.command == "distribution":
        plot_final_distribution(args.snapshots_csv, args.output, bins=args.bins)
    elif args.command == "final-opinions":
        output_path = args.output or args.snapshots_csv.with_name("final_opinions.csv")
        reduce_snapshots_csv(args.snapshots_csv, output_path)
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
    elif args.command == "extremized-heatmap":
        plot_extremized_share_heatmap(
            args.summary_csv,
            args.output,
            variant=args.variant,
            normalizer=args.normalizer,
            normalization_every=args.normalization_every,
        )
    elif args.command == "parameter-distribution":
        parameter_pairs = _comparison_parameter_pairs(args, parser)
        plot_final_opinion_histogram(
            args.snapshots_csv,
            args.output,
            parameter_pairs=parameter_pairs,
            variant=args.variant,
            normalizer=args.normalizer,
            normalization_every=args.normalization_every,
            bins=args.bins,
        )
    elif args.command == "extremized-trajectory":
        parameter_pairs = _comparison_parameter_pairs(args, parser)
        plot_extremized_share_trajectory(
            args.trajectories_csv,
            args.output,
            parameter_pairs=parameter_pairs,
            variant=args.variant,
            normalizer=args.normalizer,
            normalization_every=args.normalization_every,
        )
    return 0


def _add_parameter_arguments(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
) -> None:
    parser.add_argument("--epsilon", type=float, required=required)
    parser.add_argument("--theta", type=float, required=required)
    parser.add_argument("--variant")
    parser.add_argument("--normalizer")
    parser.add_argument("--normalization-every")


def _add_pair_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        type=float,
        default=[],
        metavar=("EPSILON", "THETA"),
        help="epsilon/theta pair to plot; repeat this option to compare pairs",
    )


def _comparison_parameter_pairs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> list[tuple[float, float]]:
    if (args.epsilon is None) != (args.theta is None):
        parser.error("--epsilon and --theta must be provided together")
    pairs = [(pair[0], pair[1]) for pair in args.pair]
    if args.epsilon is not None and args.theta is not None:
        pairs.append((args.epsilon, args.theta))
    if not pairs:
        parser.error("provide --epsilon/--theta or at least one --pair")
    return pairs


if __name__ == "__main__":
    raise SystemExit(main())
