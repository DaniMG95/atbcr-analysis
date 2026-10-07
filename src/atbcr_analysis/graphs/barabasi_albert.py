"""Barabasi-Albert graph generator."""

from __future__ import annotations

import random

from atbcr_analysis.config import BarabasiAlbertGraphConfig
from atbcr_analysis.graphs.base import Adjacency, Edge, GraphGeneratorConfig
from atbcr_analysis.graphs.utils import from_edges


def build_barabasi_albert_graph(config: GraphGeneratorConfig, rng: random.Random) -> Adjacency:
    """Build a Barabasi-Albert graph."""

    if not isinstance(config, BarabasiAlbertGraphConfig):
        raise TypeError("barabasi_albert graph requires BarabasiAlbertGraphConfig")

    edges: set[Edge] = set()
    repeated_nodes: list[int] = []
    for i in range(config.ba_m + 1):
        for j in range(i + 1, config.ba_m + 1):
            edges.add((i, j))
            repeated_nodes.extend((i, j))

    for new_node in range(config.ba_m + 1, config.n_agents):
        targets: set[int] = set()
        while len(targets) < config.ba_m:
            targets.add(rng.choice(repeated_nodes))

        for target in targets:
            edges.add((min(new_node, target), max(new_node, target)))
            repeated_nodes.extend((new_node, target))

    return from_edges(config.n_agents, edges)
