"""Metric configuration schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetricsConfig(BaseModel):
    """Settings controlling metric computation and snapshot capture."""

    model_config = ConfigDict(frozen=True)

    unbounded_extreme_cutoff: float = Field(default=1.0, ge=0)
    extremized_threshold: float | None = Field(default=None, ge=0)
    cluster_tolerance: float = Field(default=1e-3, ge=0)
    record_every: int = Field(default=500, ge=1)
    store_opinion_snapshots: bool = True
    store_interaction_events: bool = False

    @model_validator(mode="after")
    def _apply_legacy_extremized_threshold(self) -> MetricsConfig:
        if self.extremized_threshold is not None:
            object.__setattr__(
                self,
                "unbounded_extreme_cutoff",
                self.extremized_threshold,
            )
        return self
