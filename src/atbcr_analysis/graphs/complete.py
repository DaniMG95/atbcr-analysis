"""Complete graph generator."""

from __future__ import annotations

import random

from atbcr_analysis.config import CompleteGraphConfig
from atbcr_analysis.graphs.base import Adjacency, GraphGeneratorConfig
from atbcr_analysis.graphs.utils import from_edges


def build_complete_graph(config: GraphGeneratorConfig, rng: random.Random) -> Adjacency:
    """Build a graph where every pair of agents is connected."""

    if not isinstance(config, CompleteGraphConfig):
        raise TypeError("complete graph requires CompleteGraphConfig")
    return from_edges(
        config.n_agents,
        ((i, j) for i in range(config.n_agents) for j in range(i + 1, config.n_agents)),
    )
