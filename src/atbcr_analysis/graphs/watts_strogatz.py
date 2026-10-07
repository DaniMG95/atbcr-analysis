"""Watts-Strogatz graph generator."""

from __future__ import annotations

import random

from atbcr_analysis.config import WattsStrogatzGraphConfig
from atbcr_analysis.graphs.base import Adjacency, Edge, GraphGeneratorConfig
from atbcr_analysis.graphs.utils import from_edges


def build_watts_strogatz_graph(config: GraphGeneratorConfig, rng: random.Random) -> Adjacency:
    """Build a Watts-Strogatz graph."""

    if not isinstance(config, WattsStrogatzGraphConfig):
        raise TypeError("watts_strogatz graph requires WattsStrogatzGraphConfig")

    edges: set[Edge] = set()
    half_k = config.watts_k // 2
    for node in range(config.n_agents):
        for offset in range(1, half_k + 1):
            target = (node + offset) % config.n_agents
            edges.add((min(node, target), max(node, target)))

    rewired: set[Edge] = set()
    for source, target in edges:
        if rng.random() >= config.watts_rewire_probability:
            rewired.add((source, target))
            continue
        forbidden = {source, *[b if a == source else a for a, b in rewired if source in (a, b)]}
        candidates = [node for node in range(config.n_agents) if node not in forbidden]
        if not candidates:
            rewired.add((source, target))
            continue
        new_target = rng.choice(candidates)
        rewired.add((min(source, new_target), max(source, new_target)))

    return from_edges(config.n_agents, rewired)
