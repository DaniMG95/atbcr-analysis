"""Opinion initializer configuration schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atbcr_analysis.initializers import (
    BinaryConcernOpinionInitializer,
    OpinionInitializer,
    UniformOpinionInitializer,
)


class UniformOpinionInitializerConfig(BaseModel):
    """Serializable settings for uniform opinion initialization."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["uniform"] = "uniform"
    low: float = 0.0
    high: float = 1.0

    @model_validator(mode="after")
    def _validate_range(self) -> UniformOpinionInitializerConfig:
        if self.low > self.high:
            raise ValueError("uniform initializer requires low <= high")
        return self

    def build(self) -> OpinionInitializer:
        """Build the runtime initializer."""

        return UniformOpinionInitializer(low=self.low, high=self.high)


class BinaryConcernOpinionInitializerConfig(BaseModel):
    """Serializable settings for two-range concern-based initialization."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["binary_concern"] = "binary_concern"
    concern_share: float = Field(default=0.25, ge=0, le=1)
    concern_low: float = 0.75
    concern_high: float = 1.0
    non_concern_low: float = 0.0
    non_concern_high: float = 0.75

    @model_validator(mode="after")
    def _validate_ranges(self) -> BinaryConcernOpinionInitializerConfig:
        if self.concern_low > self.concern_high:
            raise ValueError("binary concern initializer requires concern_low <= concern_high")
        if self.non_concern_low > self.non_concern_high:
            raise ValueError(
                "binary concern initializer requires non_concern_low <= non_concern_high",
            )
        return self

    def build(self) -> OpinionInitializer:
        """Build the runtime initializer."""

        return BinaryConcernOpinionInitializer(
            concern_share=self.concern_share,
            concern_low=self.concern_low,
            concern_high=self.concern_high,
            non_concern_low=self.non_concern_low,
            non_concern_high=self.non_concern_high,
        )


OpinionInitializerConfig = UniformOpinionInitializerConfig | BinaryConcernOpinionInitializerConfig
