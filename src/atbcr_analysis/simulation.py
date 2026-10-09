"""Simulation engine for opinion dynamics models."""

from __future__ import annotations

import random
from dataclasses import dataclass

from atbcr_analysis.config import SimulationConfig
from atbcr_analysis.graphs import EdgeSequence, complete_graph_edges, graph_edges, generate_graph
from atbcr_analysis.metrics import RuleCounts, StepMetrics, max_abs, summarize_step
from atbcr_analysis.models import build_model
from atbcr_analysis.models.base import InteractionEvent
from atbcr_analysis.normalizers import Normalizer


@dataclass(frozen=True, slots=True)
class OpinionSnapshot:
    """Metrics and optional opinions captured at one simulation step."""

    metrics: StepMetrics
    opinions: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Result of one stochastic ATBCR run."""

    seed: int
    graph_seed: int
    opinion_seed: int
    dynamics_seed: int
    final_metrics: StepMetrics
    trajectory: tuple[OpinionSnapshot, ...]
    final_opinions: tuple[float, ...]
    normalization_events: tuple[NormalizationEvent, ...]
    interaction_events: tuple[InteractionEvent, ...]


@dataclass(frozen=True, slots=True)
class NormalizationEvent:
    """Diagnostics captured immediately around one normalization operation."""

    step: int
    normalizer: str
    max_abs_before: float
    max_abs_after: float
    scale: float | None


@dataclass(frozen=True, slots=True)
class EffectiveSeeds:
    """Concrete random seeds used by the three stochastic simulation phases."""

    graph_seed: int
    opinion_seed: int
    dynamics_seed: int


@dataclass(frozen=True, slots=True)
class SimulationRunner:
    """Run one stochastic opinion dynamics simulation."""

    config: SimulationConfig

    @classmethod
    def from_config(cls, config: SimulationConfig) -> SimulationRunner:
        """Build a runner from a serializable config."""

        return cls(config=config)

    def run(
        self,
        seed: int,
        *,
        graph_seed: int | None = None,
        opinion_seed: int | None = None,
        dynamics_seed: int | None = None,
    ) -> SimulationResult:
        """Run a configured simulation."""

        _validate_config(self.config)
        model = build_model(self.config.model, domain=self.config.domain)
        initializer = self.config.opinion_initializer.build()
        model.validate()
        effective_seeds = _resolve_seeds(
            seed,
            graph_seed=graph_seed,
            opinion_seed=opinion_seed,
            dynamics_seed=dynamics_seed,
        )
        graph_rng = random.Random(effective_seeds.graph_seed)
        opinion_rng = random.Random(effective_seeds.opinion_seed)
        dynamics_rng = random.Random(effective_seeds.dynamics_seed)
        adjacency = generate_graph(self.config.graph, graph_rng)
        edges = _simulation_edges(self.config, adjacency)
        if not edges:
            raise ValueError("graph has no edges; choose a denser graph configuration")

        opinions = initializer.initialize(self.config.graph.n_agents, opinion_rng)
        counts = RuleCounts()
        normalizer = self.config.normalization.build()
        trajectory: list[OpinionSnapshot] = []
        normalization_events: list[NormalizationEvent] = []
        interaction_events: list[InteractionEvent] = []
        last_snapshot_counts = RuleCounts()
        last_snapshot_step = 0
        _append_snapshot(
            trajectory,
            0,
            opinions,
            counts,
            RuleCounts(),
            window_size=self.config.metrics.record_every,
            config=self.config,
        )

        for step in range(1, self.config.steps + 1):
            if self.config.metrics.store_interaction_events:
                if not hasattr(model, "step_with_event"):
                    raise ValueError(
                        "configured model does not support interaction event recording",
                    )
                interaction_event = model.step_with_event(
                    step,
                    opinions,
                    adjacency,
                    edges,
                    dynamics_rng,
                )
                counts.add(interaction_event.outcome)
                if interaction_event.outcome != "inaction":
                    interaction_events.append(interaction_event)
            else:
                counts.add(model.step(opinions, adjacency, edges, dynamics_rng))
            if self.config.normalization.should_apply(step):
                event, opinions = _normalize_with_event(step, normalizer, opinions)
                normalization_events.append(event)
            if step % self.config.metrics.record_every == 0 or step == self.config.steps:
                window_counts = counts.difference(last_snapshot_counts)
                _append_snapshot(
                    trajectory,
                    step,
                    opinions,
                    counts,
                    window_counts,
                    window_size=step - last_snapshot_step,
                    config=self.config,
                )
                last_snapshot_counts = RuleCounts(
                    confidence=counts.confidence,
                    inaction=counts.inaction,
                    repulsion=counts.repulsion,
                )
                last_snapshot_step = step

        final_metrics = trajectory[-1].metrics
        return SimulationResult(
            seed=seed,
            graph_seed=effective_seeds.graph_seed,
            opinion_seed=effective_seeds.opinion_seed,
            dynamics_seed=effective_seeds.dynamics_seed,
            final_metrics=final_metrics,
            trajectory=tuple(trajectory),
            final_opinions=tuple(opinions),
            normalization_events=tuple(normalization_events),
            interaction_events=tuple(interaction_events),
        )


def simulate_atbcr(config: SimulationConfig, seed: int) -> SimulationResult:
    """Run one ATBCR simulation."""

    return SimulationRunner.from_config(config).run(seed)


def _append_snapshot(
    trajectory: list[OpinionSnapshot],
    step: int,
    opinions: list[float],
    counts: RuleCounts,
    window_counts: RuleCounts,
    *,
    window_size: int,
    config: SimulationConfig,
) -> None:
    metrics = summarize_step(
        step,
        opinions,
        counts,
        window_counts,
        window_size=window_size,
        unbounded_extreme_cutoff=config.metrics.unbounded_extreme_cutoff,
        cluster_tolerance=config.metrics.cluster_tolerance,
        domain=config.domain,
    )
    snapshot = tuple(opinions) if config.metrics.store_opinion_snapshots else None
    trajectory.append(OpinionSnapshot(metrics=metrics, opinions=snapshot))


def _simulation_edges(
    config: SimulationConfig,
    adjacency: object,
) -> EdgeSequence:
    if config.graph.kind == "complete":
        return complete_graph_edges(config.graph.n_agents)
    return graph_edges(adjacency)


def _resolve_seeds(
    seed: int,
    *,
    graph_seed: int | None,
    opinion_seed: int | None,
    dynamics_seed: int | None,
) -> EffectiveSeeds:
    seed_rng = random.Random(seed)
    return EffectiveSeeds(
        graph_seed=graph_seed if graph_seed is not None else seed_rng.randrange(2**63),
        opinion_seed=opinion_seed if opinion_seed is not None else seed_rng.randrange(2**63),
        dynamics_seed=dynamics_seed if dynamics_seed is not None else seed_rng.randrange(2**63),
    )


def _normalize_with_event(
    step: int,
    normalizer: Normalizer,
    opinions: list[float],
) -> tuple[NormalizationEvent, list[float]]:
    before = max_abs(opinions)
    scale = normalizer.scale(opinions)
    normalized = normalizer.normalize(opinions)
    after = max_abs(normalized)
    return (
        NormalizationEvent(
            step=step,
            normalizer=normalizer.name,
            max_abs_before=before,
            max_abs_after=after,
            scale=scale,
        ),
        normalized,
    )


def _validate_config(config: SimulationConfig) -> None:
    if config.steps < 1:
        raise ValueError("steps must be positive")
    if config.metrics.record_every < 1:
        raise ValueError("record_every must be positive")
    if config.metrics.unbounded_extreme_cutoff < 0:
        raise ValueError("unbounded_extreme_cutoff must be non-negative")
    if config.metrics.cluster_tolerance < 0:
        raise ValueError("cluster_tolerance must be non-negative")
