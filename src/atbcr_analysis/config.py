"""Configuration objects for ATBCR experiments."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atbcr_analysis.initializers import (
    BinaryConcernOpinionInitializer,
    OpinionInitializer,
    UniformOpinionInitializer,
)
from atbcr_analysis.normalizers import NormalizationConfig


class ModelConfig(BaseModel):
    """Serializable model selection."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)


class ATBCRModelConfig(ModelConfig):
    """Serializable parameters for the ATBCR model."""

    model_config = ConfigDict(frozen=True)

    name: Literal["atbcr"] = "atbcr"
    epsilon: float = Field(default=0.3, ge=0)
    theta: float = Field(default=0.8, ge=0)
    mu: float = Field(default=0.25, gt=0, le=0.5)

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> ATBCRModelConfig:
        if self.epsilon >= self.theta:
            raise ValueError("epsilon must be strictly smaller than theta")
        return self

    def scale_thresholds(self, factor: float) -> ATBCRModelConfig:
        """Return model parameters with scaled confidence and repulsion thresholds."""

        return ATBCRModelConfig(
            epsilon=self.epsilon * factor,
            theta=self.theta * factor,
            mu=self.mu,
        )


ModelConfigType = ATBCRModelConfig | ModelConfig


class ScenarioConfig(BaseModel):
    """Named experiment case containing the selected model configuration."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    model: ModelConfigType


class CompleteGraphConfig(BaseModel):
    """Complete graph settings."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["complete"] = "complete"
    n_agents: int = Field(default=500, gt=1)


class RingGraphConfig(BaseModel):
    """Ring graph settings."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["ring"] = "ring"
    n_agents: int = Field(default=500, gt=1)


class BarabasiAlbertGraphConfig(BaseModel):
    """Barabasi-Albert graph settings."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["barabasi_albert"] = "barabasi_albert"
    n_agents: int = Field(default=500, gt=1)
    ba_m: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _validate_m(self) -> BarabasiAlbertGraphConfig:
        if self.ba_m >= self.n_agents:
            raise ValueError("ba_m must be smaller than n_agents")
        return self


class ErdosRenyiGraphConfig(BaseModel):
    """Erdos-Renyi graph settings."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["erdos_renyi"] = "erdos_renyi"
    n_agents: int = Field(default=500, gt=1)
    er_probability: float = Field(default=0.02, ge=0, le=1)


class WattsStrogatzGraphConfig(BaseModel):
    """Watts-Strogatz graph settings."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["watts_strogatz"] = "watts_strogatz"
    n_agents: int = Field(default=500, gt=1)
    watts_k: int = Field(default=6, ge=2)
    watts_rewire_probability: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def _validate_k(self) -> WattsStrogatzGraphConfig:
        if self.watts_k >= self.n_agents or self.watts_k % 2 != 0:
            raise ValueError("watts_k must be an even number in [2, n_agents)")
        return self


GraphConfig = (
    CompleteGraphConfig
    | RingGraphConfig
    | BarabasiAlbertGraphConfig
    | ErdosRenyiGraphConfig
    | WattsStrogatzGraphConfig
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
                "model": _scale_model_thresholds(scenario.model, variant.threshold_scale),
                "variant": variant.name,
                "domain": variant.domain,
                "opinion_initializer": variant.initializer,
            },
        )


class MonteCarloConfig(BaseModel):
    """Monte Carlo execution settings."""

    model_config = ConfigDict(frozen=True)

    runs: int = Field(default=20, ge=1)
    seed: int = 7
    graph_seed: int | None = None
    opinion_seed: int | None = None
    dynamics_seed: int | None = None
    workers: int = Field(default=1, ge=1)


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


def _scale_model_thresholds(model: ModelConfigType, factor: float) -> ModelConfigType:
    if isinstance(model, ATBCRModelConfig):
        return model.scale_thresholds(factor)
    if factor != 1.0:
        raise ValueError("threshold scaling is only supported by ATBCRModelConfig")
    return model
