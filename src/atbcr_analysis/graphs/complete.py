"""Complete graph generator."""

from __future__ import annotations

import random

from atbcr_analysis.config import CompleteGraphConfig
from atbcr_analysis.graphs.base import Adjacency, GraphGeneratorConfig
from atbcr_analysis.graphs.utils import complete_graph_adjacency


def build_complete_graph(config: GraphGeneratorConfig, rng: random.Random) -> Adjacency:
    """Build a graph where every pair of agents is connected."""

    if not isinstance(config, CompleteGraphConfig):
        raise TypeError("complete graph requires CompleteGraphConfig")
    return complete_graph_adjacency(config.n_agents)
