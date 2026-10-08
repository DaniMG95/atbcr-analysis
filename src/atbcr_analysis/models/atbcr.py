"""ATBCR model implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass

from atbcr_analysis.config import ATBCRModelConfig
from atbcr_analysis.graphs import Adjacency, Edge
from atbcr_analysis.models.base import InteractionOutcome


@dataclass(frozen=True, slots=True)
class ATBCRModel:
    """Agent-independent time-based bounded confidence and repulsion model."""

    config: ATBCRModelConfig
    domain: str = "bounded_01"

    @property
    def name(self) -> str:
        return "atbcr"

    def step(
        self,
        opinions: list[float],
        adjacency: Adjacency,
        edges: tuple[Edge, ...],
        rng: random.Random,
    ) -> InteractionOutcome:
        if not edges:
            raise ValueError("graph has no edges; choose a denser graph configuration")
        first, second = rng.choice(edges)
        return self._update_pair(opinions, first, second)

    def _update_pair(self, opinions: list[float], first: int, second: int) -> InteractionOutcome:
        xi = opinions[first]
        xj = opinions[second]
        distance = abs(xi - xj)
        delta = self.config.mu * (xj - xi)

        if distance < self.config.epsilon:
            opinions[first] = self._clip(xi + delta)
            opinions[second] = self._clip(xj - delta)
            return "confidence"
        if distance > self.config.theta:
            opinions[first] = self._clip(xi - delta)
            opinions[second] = self._clip(xj + delta)
            return "repulsion"
        return "inaction"

    def validate(self) -> None:
        if self.config.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.config.theta < 0:
            raise ValueError("theta must be non-negative")
        if not 0 < self.config.mu <= 0.5:
            raise ValueError("mu must be in (0, 0.5]")
        if self.config.epsilon >= self.config.theta:
            raise ValueError("epsilon must be strictly smaller than theta")
        if self.domain not in {"bounded_01", "bounded_m11", "unbounded"}:
            raise ValueError("domain must be bounded_01, bounded_m11, or unbounded")

    def _clip(self, value: float) -> float:
        if self.domain == "bounded_01":
            return min(1.0, max(0.0, value))
        if self.domain == "bounded_m11":
            return min(1.0, max(-1.0, value))
        return value
