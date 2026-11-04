"""ATBCR model implementation."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from atbcr_analysis.config import ATBCRModelConfig
from atbcr_analysis.graphs import Adjacency, Edge
from atbcr_analysis.models.base import InteractionEvent, InteractionOutcome


@dataclass(slots=True)
class ATBCRModel:
    """Agent-independent time-based bounded confidence and repulsion model."""

    config: ATBCRModelConfig
    domain: str = "bounded_01"
    _completed_steps: int = field(default=0, init=False, repr=False)
    _initial_max_abs: float | None = field(default=None, init=False, repr=False)
    _effective_epsilon: float = field(default=0.0, init=False, repr=False)
    _effective_theta: float = field(default=0.0, init=False, repr=False)

    @property
    def name(self) -> str:
        return "atbcr"

    def step(
        self,
        opinions: list[float],
        adjacency: Adjacency,
        edges: Sequence[Edge],
        rng: random.Random,
    ) -> InteractionOutcome:
        if not edges:
            raise ValueError("graph has no edges; choose a denser graph configuration")
        first, second = rng.choice(edges)
        return self._update_pair(opinions, first, second)

    def step_with_event(
        self,
        step: int,
        opinions: list[float],
        adjacency: Adjacency,
        edges: Sequence[Edge],
        rng: random.Random,
    ) -> InteractionEvent:
        """Apply one step and return the full interaction event."""

        if not edges:
            raise ValueError("graph has no edges; choose a denser graph configuration")
        first, second = rng.choice(edges)
        first_before = opinions[first]
        second_before = opinions[second]
        outcome = self._update_pair(opinions, first, second)
        return InteractionEvent(
            step=step,
            first_agent=first,
            second_agent=second,
            outcome=outcome,
            first_opinion_before=first_before,
            second_opinion_before=second_before,
            first_opinion_after=opinions[first],
            second_opinion_after=opinions[second],
        )

    def _update_pair(self, opinions: list[float], first: int, second: int) -> InteractionOutcome:
        epsilon, theta = self._thresholds(opinions)
        xi = opinions[first]
        xj = opinions[second]
        distance = abs(xi - xj)
        delta = self.config.mu * (xj - xi)

        if distance < epsilon:
            self._store_updated_pair(opinions, first, second, xi + delta, xj - delta)
            return "confidence"
        if distance > theta:
            self._store_updated_pair(opinions, first, second, xi - delta, xj + delta)
            return "repulsion"
        return "inaction"

    def initialize(self, opinions: Sequence[float]) -> None:
        """Capture the run's initial scale and reset the effective thresholds."""

        if self.config.threshold_mode == "fixed":
            return
        initial_max_abs = self._finite_max_abs(opinions, context="initial opinions")
        self._initial_max_abs = initial_max_abs
        self._effective_epsilon = self.config.epsilon
        self._effective_theta = self.config.theta
        self._completed_steps = 0

    @property
    def effective_thresholds(self) -> tuple[float, float]:
        """Confidence and repulsion thresholds used by the latest interaction."""

        if self.config.threshold_mode == "fixed" or self._initial_max_abs is None:
            return self.config.epsilon, self.config.theta
        return self._effective_epsilon, self._effective_theta

    @property
    def initial_max_abs(self) -> float | None:
        """Maximum absolute opinion captured once at the start of the run."""

        return self._initial_max_abs

    def _thresholds(self, opinions: Sequence[float]) -> tuple[float, float]:
        if self.config.threshold_mode == "fixed":
            return self.config.epsilon, self.config.theta
        if self._initial_max_abs is None:
            self.initialize(opinions)
        if self._completed_steps and (
            self._completed_steps % self.config.threshold_update_every == 0
        ):
            self._refresh_dynamic_thresholds(opinions)
        self._completed_steps += 1
        return self._effective_epsilon, self._effective_theta

    def _refresh_dynamic_thresholds(self, opinions: Sequence[float]) -> None:
        if self._initial_max_abs is None:
            raise RuntimeError("dynamic thresholds must be initialized before they are refreshed")
        current_max_abs = self._finite_max_abs(
            opinions,
            context=f"opinions at time step {self._completed_steps}",
        )
        # An all-zero initial state has no meaningful relative scale. Keeping
        # the domain-adjusted initial thresholds is explicit and avoids 0/0.
        scale = 1.0 if self._initial_max_abs == 0.0 else current_max_abs / self._initial_max_abs
        epsilon = self.config.epsilon * scale
        theta = self.config.theta * scale
        if scale < 0 or not all(math.isfinite(value) for value in (scale, epsilon, theta)):
            raise FloatingPointError(
                "dynamic threshold overflow at time step "
                f"{self._completed_steps}: scale={scale!r}, "
                f"epsilon={epsilon!r}, theta={theta!r}",
            )
        self._effective_epsilon = epsilon
        self._effective_theta = theta

    @staticmethod
    def _finite_max_abs(opinions: Sequence[float], *, context: str) -> float:
        if not opinions:
            raise ValueError("opinions cannot be empty")
        maximum = max(abs(opinion) for opinion in opinions)
        if not math.isfinite(maximum):
            raise FloatingPointError(f"non-finite value found in {context}: max_abs={maximum!r}")
        return maximum

    def _store_updated_pair(
        self,
        opinions: list[float],
        first: int,
        second: int,
        first_value: float,
        second_value: float,
    ) -> None:
        first_value = self._clip(first_value)
        second_value = self._clip(second_value)
        if self.config.threshold_mode == "max_abs" and not all(
            math.isfinite(value) for value in (first_value, second_value)
        ):
            raise FloatingPointError(
                "opinion overflow during dynamic-threshold simulation at time step "
                f"{self._completed_steps}: values=({first_value!r}, {second_value!r})",
            )
        opinions[first] = first_value
        opinions[second] = second_value

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
