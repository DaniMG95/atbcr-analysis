"""Experiment variant configuration schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from atbcr_analysis.config.initializers import (
    OpinionInitializerConfig,
    UniformOpinionInitializerConfig,
)
from atbcr_analysis.normalizers import NormalizationConfig


class VariantConfig(BaseModel):
    """Settings for an experiment variant over the same scenario and seeds."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    domain: Literal["bounded_01", "bounded_m11", "unbounded"]
    threshold_scale: float = Field(default=1.0, gt=0)
    initializer: OpinionInitializerConfig = UniformOpinionInitializerConfig()
    normalizations: tuple[NormalizationConfig, ...] = (NormalizationConfig(),)


DEFAULT_VARIANTS: tuple[VariantConfig, ...] = (
    VariantConfig(
        name="baseline_01",
        domain="bounded_01",
        threshold_scale=1.0,
        initializer=UniformOpinionInitializerConfig(low=0.0, high=1.0),
    ),
    VariantConfig(
        name="bounded_m11",
        domain="bounded_m11",
        threshold_scale=2.0,
        initializer=UniformOpinionInitializerConfig(low=-1.0, high=1.0),
    ),
    VariantConfig(
        name="unbounded",
        domain="unbounded",
        threshold_scale=2.0,
        initializer=UniformOpinionInitializerConfig(low=-1.0, high=1.0),
    ),
)
