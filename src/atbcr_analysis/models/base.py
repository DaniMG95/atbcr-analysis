"""Base contracts for opinion dynamics models."""

from __future__ import annotations

import random
from typing import Literal, Protocol

from atbcr_analysis.graphs import Adjacency, Edge

InteractionOutcome = Literal["confidence", "inaction", "repulsion"]


class OpinionDynamicsModel(Protocol):
    """Contract implemented by every opinion dynamics model."""

    @property
    def name(self) -> str:
        """Human-readable model name."""

    def step(
        self,
        opinions: list[float],
        adjacency: Adjacency,
        edges: tuple[Edge, ...],
        rng: random.Random,
    ) -> InteractionOutcome:
        """Apply one model-specific simulation step in place and return the applied rule."""

    def validate(self) -> None:
        """Validate model parameters before a simulation starts."""
