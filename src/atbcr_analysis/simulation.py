"""Simulation engine for opinion dynamics models."""

from __future__ import annotations

import random
from dataclasses import dataclass

from atbcr_analysis.config import SimulationConfig
from atbcr_analysis.graphs import graph_edges, generate_graph
from atbcr_analysis.metrics import RuleCounts, StepMetrics, summarize_step
from atbcr_analysis.models import build_model


@dataclass(frozen=True, slots=True)
class OpinionSnapshot:
    """Metrics and optional opinions captured at one simulation step."""

    metrics: StepMetrics
    opinions: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Result of one stochastic ATBCR run."""

    seed: int
    final_metrics: StepMetrics
    trajectory: tuple[OpinionSnapshot, ...]
    final_opinions: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SimulationRunner:
    """Run one stochastic opinion dynamics simulation."""

    config: SimulationConfig

    @classmethod
    def from_config(cls, config: SimulationConfig) -> SimulationRunner:
        """Build a runner from a serializable config."""

        return cls(config=config)

    def run(self, seed: int) -> SimulationResult:
        """Run a configured simulation."""

        _validate_config(self.config)
        model = build_model(self.config.model, domain=self.config.domain)
        initializer = self.config.opinion_initializer.build()
        model.validate()
        rng = random.Random(seed)
        adjacency = generate_graph(self.config.graph, rng)
        edges = graph_edges(adjacency)
        if not edges:
            raise ValueError("graph has no edges; choose a denser graph configuration")

        opinions = initializer.initialize(self.config.graph.n_agents, rng)
        counts = RuleCounts()
        normalizer = self.config.normalization.build()
        trajectory: list[OpinionSnapshot] = []
        _append_snapshot(trajectory, 0, opinions, counts, self.config)

        for step in range(1, self.config.steps + 1):
            counts.add(model.step(opinions, adjacency, edges, rng))
            if self.config.normalization.should_apply(step):
                opinions = normalizer.normalize(opinions)
            if step % self.config.metrics.record_every == 0 or step == self.config.steps:
                _append_snapshot(trajectory, step, opinions, counts, self.config)

        final_metrics = trajectory[-1].metrics
        return SimulationResult(
            seed=seed,
            final_metrics=final_metrics,
            trajectory=tuple(trajectory),
            final_opinions=tuple(opinions),
        )


def simulate_atbcr(config: SimulationConfig, seed: int) -> SimulationResult:
    """Run one ATBCR simulation."""

    return SimulationRunner.from_config(config).run(seed)


def _append_snapshot(
    trajectory: list[OpinionSnapshot],
    step: int,
    opinions: list[float],
    counts: RuleCounts,
    config: SimulationConfig,
) -> None:
    metrics = summarize_step(
        step,
        opinions,
        counts,
        extremized_threshold=config.metrics.extremized_threshold,
        cluster_tolerance=config.metrics.cluster_tolerance,
        domain=config.domain,
    )
    snapshot = tuple(opinions) if config.metrics.store_opinion_snapshots else None
    trajectory.append(OpinionSnapshot(metrics=metrics, opinions=snapshot))


def _validate_config(config: SimulationConfig) -> None:
    if config.steps < 1:
        raise ValueError("steps must be positive")
    if config.metrics.record_every < 1:
        raise ValueError("record_every must be positive")
    if config.metrics.extremized_threshold < 0:
        raise ValueError("extremized_threshold must be non-negative")
    if config.metrics.cluster_tolerance < 0:
        raise ValueError("cluster_tolerance must be non-negative")
