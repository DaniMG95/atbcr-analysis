"""Simulation configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from atbcr_analysis.config.graphs import CompleteGraphConfig, GraphConfig
from atbcr_analysis.config.initializers import (
    OpinionInitializerConfig,
    UniformOpinionInitializerConfig,
)
from atbcr_analysis.config.metrics import MetricsConfig
from atbcr_analysis.config.models import ModelConfigType, scale_model_thresholds
from atbcr_analysis.config.scenarios import ScenarioConfig
from atbcr_analysis.config.variants import VariantConfig
from atbcr_analysis.normalizers import NormalizationConfig


class SimulationConfig(BaseModel):
    """Single-run simulation settings."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    model: ModelConfigType
    graph: GraphConfig = Field(default_factory=CompleteGraphConfig)
    variant: str = "baseline_01"
    domain: str = "bounded_01"
    steps: int = Field(default=13_500, ge=1)
    metrics: MetricsConfig = MetricsConfig()
    opinion_initializer: OpinionInitializerConfig = UniformOpinionInitializerConfig()
    normalization: NormalizationConfig = NormalizationConfig()

    def with_scenario(self, scenario: ScenarioConfig) -> SimulationConfig:
        """Return a copy with a different scenario."""

        return self.model_copy(update={"name": scenario.name, "model": scenario.model})

    def with_variant(self, variant: VariantConfig, scenario: ScenarioConfig) -> SimulationConfig:
        """Return a copy with variant settings applied."""

        return self.model_copy(
            update={
                "name": scenario.name,
                "model": scale_model_thresholds(scenario.model, variant.threshold_scale),
                "variant": variant.name,
                "domain": variant.domain,
                "opinion_initializer": variant.initializer,
            },
        )
