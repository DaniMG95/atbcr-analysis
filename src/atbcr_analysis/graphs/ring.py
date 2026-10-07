"""Ring graph generator."""

from __future__ import annotations

import random

from atbcr_analysis.config import RingGraphConfig
from atbcr_analysis.graphs.base import Adjacency, GraphGeneratorConfig
from atbcr_analysis.graphs.utils import from_edges


def build_ring_graph(config: GraphGeneratorConfig, rng: random.Random) -> Adjacency:
    """Build a ring graph."""

    if not isinstance(config, RingGraphConfig):
        raise TypeError("ring graph requires RingGraphConfig")
    return from_edges(
        config.n_agents,
        ((i, (i + 1) % config.n_agents) for i in range(config.n_agents)),
    )
