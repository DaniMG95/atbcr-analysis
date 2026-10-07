"""Erdos-Renyi graph generator."""

from __future__ import annotations

import random

from atbcr_analysis.config import ErdosRenyiGraphConfig
from atbcr_analysis.graphs.base import Adjacency, GraphGeneratorConfig
from atbcr_analysis.graphs.utils import from_edges


def build_erdos_renyi_graph(config: GraphGeneratorConfig, rng: random.Random) -> Adjacency:
    """Build an Erdos-Renyi graph."""

    if not isinstance(config, ErdosRenyiGraphConfig):
        raise TypeError("erdos_renyi graph requires ErdosRenyiGraphConfig")
    edges = (
        (i, j)
        for i in range(config.n_agents)
        for j in range(i + 1, config.n_agents)
        if rng.random() < config.er_probability
    )
    return from_edges(config.n_agents, edges)
