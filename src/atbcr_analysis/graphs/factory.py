"""Factory and registry for graph generators."""

from __future__ import annotations

import random

from atbcr_analysis.config import GraphConfig
from atbcr_analysis.graphs.barabasi_albert import build_barabasi_albert_graph
from atbcr_analysis.graphs.base import Adjacency, GraphBuilder, GraphGeneratorConfig
from atbcr_analysis.graphs.complete import build_complete_graph
from atbcr_analysis.graphs.erdos_renyi import build_erdos_renyi_graph
from atbcr_analysis.graphs.ring import build_ring_graph
from atbcr_analysis.graphs.watts_strogatz import build_watts_strogatz_graph


class GraphFactory:
    """Registry-backed factory for graph generators."""

    _builders: dict[str, GraphBuilder] = {}

    @classmethod
    def register(cls, kind: str, builder: GraphBuilder) -> None:
        normalized = _normalize_kind(kind)
        if not normalized:
            raise ValueError("graph kind cannot be empty")
        cls._builders[normalized] = builder

    @classmethod
    def create(cls, config: GraphGeneratorConfig, rng: random.Random) -> Adjacency:
        normalized = _normalize_kind(config.kind)
        if normalized not in cls._builders:
            raise ValueError(f"Unsupported graph kind: {config.kind}")
        return cls._builders[normalized](config, rng)

    @classmethod
    def available_graphs(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._builders))


def generate_graph(config: GraphConfig, rng: random.Random) -> Adjacency:
    """Generate an undirected graph as adjacency lists."""

    return GraphFactory.create(config, rng)


def register_graph(kind: str, builder: GraphBuilder) -> None:
    """Register a graph builder in the package-level factory."""

    GraphFactory.register(kind, builder)


def _normalize_kind(kind: str) -> str:
    return kind.lower().replace("-", "_")


GraphFactory.register("complete", build_complete_graph)
GraphFactory.register("ring", build_ring_graph)
GraphFactory.register("erdos_renyi", build_erdos_renyi_graph)
GraphFactory.register("barabasi_albert", build_barabasi_albert_graph)
GraphFactory.register("watts_strogatz", build_watts_strogatz_graph)
