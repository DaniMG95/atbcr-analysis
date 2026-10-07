"""Helpers for building paired experiment grids."""

from __future__ import annotations

from atbcr_analysis.config import (
    DEFAULT_VARIANTS,
    ScenarioConfig,
    SimulationConfig,
    VariantConfig,
)
from atbcr_analysis.normalizers import NormalizationConfig


def build_paired_grid(
    base_config: SimulationConfig,
    *,
    scenarios: tuple[ScenarioConfig, ...],
    variants: tuple[VariantConfig, ...] = DEFAULT_VARIANTS,
    normalizations: tuple[NormalizationConfig, ...] = (NormalizationConfig(),),
) -> tuple[SimulationConfig, ...]:
    """Build configs intended to be run with the same Monte Carlo seed sequence."""

    configs: list[SimulationConfig] = []
    for scenario in scenarios:
        for variant in variants:
            variant_config = base_config.with_variant(variant, scenario)
            for normalization in normalizations:
                configs.append(variant_config.model_copy(update={"normalization": normalization}))
    return tuple(configs)
