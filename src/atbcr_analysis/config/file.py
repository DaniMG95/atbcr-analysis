"""YAML file configuration schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atbcr_analysis.config.graphs import CompleteGraphConfig, GraphConfig
from atbcr_analysis.config.initializers import OpinionInitializerConfig
from atbcr_analysis.config.metrics import MetricsConfig
from atbcr_analysis.config.monte_carlo import MonteCarloConfig
from atbcr_analysis.config.scenarios import (
    ScenarioConfig,
    ScenarioGridConfig,
    ScenarioParametersConfig,
)
from atbcr_analysis.config.variants import DEFAULT_VARIANTS, VariantConfig
from atbcr_analysis.normalizers import NormalizationConfig


class RunConfig(BaseModel):
    """Validated configuration for one experiment run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenarios: dict[str, ScenarioParametersConfig] | None = None
    scenario_grid: ScenarioGridConfig | None = None
    variants: tuple[VariantConfig, ...] = DEFAULT_VARIANTS
    graph: GraphConfig = Field(default_factory=CompleteGraphConfig)
    steps: int = Field(default=13_500, ge=1)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    opinion_initializer: OpinionInitializerConfig | None = None
    normalizations: tuple[NormalizationConfig, ...] = (NormalizationConfig(),)
    output_dir: Path = Path("runs/experiment")
    show_progress: bool = True

    @model_validator(mode="after")
    def _validate_scenario_source(self) -> RunConfig:
        if self.scenarios is not None and self.scenario_grid is not None:
            raise ValueError("Use either 'scenarios' or 'scenario_grid', not both")
        if self.scenarios is None and self.scenario_grid is None:
            raise ValueError("Configure either 'scenarios' or 'scenario_grid'")
        if self.scenario_grid is not None:
            self.scenario_grid.scenarios()
        return self

    @property
    def resolved_scenarios(self) -> tuple[ScenarioConfig, ...]:
        if self.scenario_grid is not None:
            return self.scenario_grid.scenarios()
        if self.scenarios is None:
            raise ValueError("Configure either 'scenarios' or 'scenario_grid'")
        return _named_scenarios(self.scenarios)

    @property
    def resolved_variants(self) -> tuple[VariantConfig, ...]:
        if self.opinion_initializer is None:
            return self.variants
        return tuple(
            variant.model_copy(update={"initializer": self.opinion_initializer})
            for variant in self.variants
        )


class DefaultsConfig(BaseModel):
    """Defaults shared by experiment runs in a YAML file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variants: tuple[VariantConfig, ...] | None = None
    graph: GraphConfig | None = None
    steps: int | None = Field(default=None, ge=1)
    metrics: MetricsConfig | None = None
    monte_carlo: MonteCarloConfig | None = None
    opinion_initializer: OpinionInitializerConfig | None = None
    normalizations: tuple[NormalizationConfig, ...] | None = None
    show_progress: bool | None = None

    def values(self) -> dict[str, Any]:
        return self.model_dump(
            exclude_none=True,
            exclude_unset=True,
            exclude_computed_fields=True,
        )


class ExperimentConfig(DefaultsConfig):
    """One experiment entry in a YAML file."""

    scenarios: dict[str, ScenarioParametersConfig] | None = None
    scenario_grid: ScenarioGridConfig | None = None
    output_dir: Path | None = None

    def with_defaults(self, defaults: DefaultsConfig) -> RunConfig:
        values = _merge_mapping(defaults.values(), self.values())
        return RunConfig.model_validate(values)


class FileConfig(BaseModel):
    """Validated YAML file containing one or more experiment runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    experiments: tuple[ExperimentConfig, ...]

    @model_validator(mode="after")
    def _validate_experiments(self) -> FileConfig:
        if not self.experiments:
            raise ValueError("YAML 'experiments' must be a non-empty list")
        return self

    @property
    def runs(self) -> tuple[RunConfig, ...]:
        return tuple(
            experiment.with_defaults(self.defaults)
            for experiment in self.experiments
        )

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> FileConfig:
        return cls.model_validate(config)


def _named_scenarios(
    scenarios: dict[str, ScenarioParametersConfig],
) -> tuple[ScenarioConfig, ...]:
    return tuple(
        ScenarioConfig(name=name, **parameters.model_dump())
        for name, parameters in scenarios.items()
    )


def _merge_mapping(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in override.items():
        default_value = merged.get(key)
        if isinstance(default_value, dict) and isinstance(value, dict):
            merged[key] = _merge_mapping(default_value, value)
            continue
        merged[key] = value
    return merged
