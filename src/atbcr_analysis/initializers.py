"""Opinion initializers for simulations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


class OpinionInitializer(Protocol):
    """Build the starting opinions for a simulation."""

    def initialize(self, n_agents: int, rng: random.Random) -> list[float]:
        """Return one opinion for each agent."""


@dataclass(frozen=True, slots=True)
class UniformOpinionInitializer:
    """Initialize opinions uniformly inside a configurable interval."""

    low: float = 0.0
    high: float = 1.0

    def initialize(self, n_agents: int, rng: random.Random) -> list[float]:
        _validate_range(self.low, self.high)
        return [rng.uniform(self.low, self.high) for _ in range(n_agents)]


@dataclass(frozen=True, slots=True)
class BinaryConcernOpinionInitializer:
    """Initialize opinions from two intervals split by a concern threshold."""

    concern_share: float = 0.25
    concern_low: float = 0.75
    concern_high: float = 1.0
    non_concern_low: float = 0.0
    non_concern_high: float = 0.75

    def initialize(self, n_agents: int, rng: random.Random) -> list[float]:
        if not 0 <= self.concern_share <= 1:
            raise ValueError("concern_share must be in [0, 1]")
        _validate_range(self.concern_low, self.concern_high)
        _validate_range(self.non_concern_low, self.non_concern_high)
        return [
            rng.uniform(self.concern_low, self.concern_high)
            if rng.random() < self.concern_share
            else rng.uniform(self.non_concern_low, self.non_concern_high)
            for _ in range(n_agents)
        ]


def _validate_range(low: float, high: float) -> None:
    if low > high:
        raise ValueError("opinion ranges must satisfy low <= high")
