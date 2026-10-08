"""Shared graph utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import sqrt

from atbcr_analysis.graphs.base import Adjacency, Edge, EdgeSequence


def graph_edges(adjacency: Sequence[Sequence[int]]) -> list[Edge]:
    """Return each undirected graph edge once."""

    return [(i, j) for i, neighbors in enumerate(adjacency) for j in neighbors if i < j]


@dataclass(frozen=True, slots=True)
class CompleteGraphEdges(EdgeSequence):
    """Lazy sequence of all undirected complete-graph edges."""

    n_agents: int

    def __len__(self) -> int:
        return self.n_agents * (self.n_agents - 1) // 2

    def __getitem__(self, index: int) -> Edge:
        edge_count = len(self)
        if index < 0:
            index += edge_count
        if index < 0 or index >= edge_count:
            raise IndexError(index)

        discriminant = (2 * self.n_agents - 1) ** 2 - 8 * index
        row = int((2 * self.n_agents - 1 - sqrt(discriminant)) // 2)
        while _complete_graph_row_start(row + 1, self.n_agents) <= index:
            row += 1
        while _complete_graph_row_start(row, self.n_agents) > index:
            row -= 1
        offset = index - _complete_graph_row_start(row, self.n_agents)
        return row, row + 1 + offset


def complete_graph_edges(n_agents: int) -> CompleteGraphEdges:
    """Return a lazy complete-graph edge sequence."""

    if n_agents < 2:
        raise ValueError("complete graph requires at least two agents")
    return CompleteGraphEdges(n_agents=n_agents)


def from_edges(n_agents: int, edges: Iterable[Edge]) -> Adjacency:
    """Build an undirected adjacency list from an edge iterable."""

    adjacency: list[set[int]] = [set() for _ in range(n_agents)]
    for source, target in edges:
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return [tuple(sorted(neighbors)) for neighbors in adjacency]


def _complete_graph_row_start(row: int, n_agents: int) -> int:
    return row * (2 * n_agents - row - 1) // 2
