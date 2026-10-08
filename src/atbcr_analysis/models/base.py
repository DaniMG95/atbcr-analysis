"""Base contracts for opinion dynamics models."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from atbcr_analysis.graphs import Adjacency, Edge

InteractionOutcome = Literal["confidence", "inaction", "repulsion"]


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    """One recorded pair interaction with before/after opinions."""

    step: int
    first_agent: int
    second_agent: int
    outcome: InteractionOutcome
    first_opinion_before: float
    second_opinion_before: float
    first_opinion_after: float
    second_opinion_after: float


class OpinionDynamicsModel(Protocol):
    """Contract implemented by every opinion dynamics model."""

    @property
    def name(self) -> str:
        """Human-readable model name."""

    def step(
        self,
        opinions: list[float],
        adjacency: Adjacency,
        edges: Sequence[Edge],
        rng: random.Random,
    ) -> InteractionOutcome:
        """Apply one model-specific simulation step in place and return the applied rule."""

    def validate(self) -> None:
        """Validate model parameters before a simulation starts."""
