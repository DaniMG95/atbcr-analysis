"""Plotting helpers and command-line support."""

from atbcr_analysis.plots.comparative import (
    AggregatePoint,
    ComparativeSeries,
    LinearFit,
    aggregate_trajectories,
    fit_replica_trajectories,
    plot_comparative_trajectories,
    read_comparative_trajectories,
    write_linear_fits_csv,
)
from atbcr_analysis.plots.plots import (
    ExtremizedShareGrid,
    SnapshotSeries,
    _build_opinion_plot_title,
    _final_opinions_for_parameters,
    _final_opinions_for_parameter_pairs,
    _mean_extremized_share_for_parameters,
    _mean_extremized_shares_for_parameters,
    _read_extremized_share_grid,
    plot_extremized_share_heatmap,
    plot_extremized_share_trajectory,
    plot_final_distribution,
    plot_final_opinion_histogram,
    plot_metric_trajectories,
    plot_opinion_trajectories,
)

__all__ = [
    "AggregatePoint",
    "ComparativeSeries",
    "ExtremizedShareGrid",
    "LinearFit",
    "SnapshotSeries",
    "aggregate_trajectories",
    "fit_replica_trajectories",
    "plot_comparative_trajectories",
    "plot_extremized_share_heatmap",
    "plot_extremized_share_trajectory",
    "plot_final_distribution",
    "plot_final_opinion_histogram",
    "plot_metric_trajectories",
    "plot_opinion_trajectories",
    "read_comparative_trajectories",
    "write_linear_fits_csv",
]
