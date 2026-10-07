"""Normalization policies applied by the simulation runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class Normalizer(Protocol):
    """Transform the full opinion vector independently of the model."""

    @property
    def name(self) -> str:
        """Human-readable normalizer name."""

    def normalize(self, opinions: list[float]) -> list[float]:
        """Return normalized opinions."""


@dataclass(frozen=True, slots=True)
class NoNormalizer:
    """Leave opinions unchanged."""

    @property
    def name(self) -> str:
        return "none"

    def normalize(self, opinions: list[float]) -> list[float]:
        return opinions


@dataclass(frozen=True, slots=True)
class MaxAbsNormalizer:
    """Normalize by max(abs(x))."""

    @property
    def name(self) -> str:
        return "max_abs"

    def normalize(self, opinions: list[float]) -> list[float]:
        scale = max(abs(opinion) for opinion in opinions)
        if scale == 0:
            return opinions
        return [opinion / scale for opinion in opinions]


@dataclass(frozen=True, slots=True)
class SignedLogMaxAbsNormalizer:
    """Apply signed log compression and then max-abs normalization."""

    @property
    def name(self) -> str:
        return "signed_log_max_abs"

    def normalize(self, opinions: list[float]) -> list[float]:
        compressed = [
            math.copysign(math.log1p(abs(opinion)), opinion)
            for opinion in opinions
        ]
        return MaxAbsNormalizer().normalize(compressed)


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Serializable normalization settings."""

    kind: str = "none"
    every: int | None = None

    def build(self) -> Normalizer:
        normalized = self.kind.lower().replace("-", "_")
        if normalized == "none":
            return NoNormalizer()
        if normalized == "max_abs":
            return MaxAbsNormalizer()
        if normalized in {"signed_log", "signed_log_max_abs"}:
            return SignedLogMaxAbsNormalizer()
        raise ValueError(f"Unsupported normalizer: {self.kind}")

    def should_apply(self, step: int) -> bool:
        return self.kind != "none" and self.every is not None and self.every > 0 and step % self.every == 0
