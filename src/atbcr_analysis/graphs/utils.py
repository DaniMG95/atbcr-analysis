"""Shared graph utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from atbcr_analysis.graphs.base import Adjacency, Edge


def graph_edges(adjacency: Sequence[Sequence[int]]) -> list[Edge]:
    """Return each undirected graph edge once."""

    return [(i, j) for i, neighbors in enumerate(adjacency) for j in neighbors if i < j]


def from_edges(n_agents: int, edges: Iterable[Edge]) -> Adjacency:
    """Build an undirected adjacency list from an edge iterable."""

    adjacency: list[set[int]] = [set() for _ in range(n_agents)]
    for source, target in edges:
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return [tuple(sorted(neighbors)) for neighbors in adjacency]
