"""Base contracts for graph generators."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Protocol

Adjacency = Sequence[Sequence[int]]
Edge = tuple[int, int]
EdgeSequence = Sequence[Edge]


class GraphGeneratorConfig(Protocol):
    """Minimum config contract needed by the graph factory."""

    kind: str
    n_agents: int


GraphBuilder = Callable[[GraphGeneratorConfig, random.Random], Adjacency]
