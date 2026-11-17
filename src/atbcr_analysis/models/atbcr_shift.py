"""ATBCR model with an alternative individual-shift branch."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from atbcr_analysis.config import ATBCRShiftModelConfig
from atbcr_analysis.graphs import Adjacency, Edge
from atbcr_analysis.models.atbcr import ATBCRModel
from atbcr_analysis.models.base import InteractionOutcome


@dataclass(slots=True)
class ATBCRShiftModel:
    """Choose exactly one ATBCR interaction or one clipped individual shift per step."""

    config: ATBCRShiftModelConfig
    domain: str = "bounded_m11"
    _atbcr: ATBCRModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._atbcr = ATBCRModel(config=self.config, domain=self.domain)

    @property
    def name(self) -> str:
        return "atbcr_shift"

    def initialize(self, opinions: Sequence[float]) -> None:
        self._atbcr.initialize(opinions)

    def step(
        self,
        opinions: list[float],
        adjacency: Adjacency,
        edges: Sequence[Edge],
        rng: random.Random,
    ) -> InteractionOutcome:
        if self.config.p == 1.0 or (self.config.p != 0.0 and rng.random() < self.config.p):
            return self._atbcr.step(opinions, adjacency, edges, rng)

        agent = rng.randrange(len(opinions))
        opinions[agent] = min(1.0, max(-1.0, opinions[agent] + self.config.shift_lambda))
        return "shift"

    def validate(self) -> None:
        if self.domain != "bounded_m11":
            raise ValueError("atbcr_shift requires the bounded_m11 domain")
        self._atbcr.validate()
