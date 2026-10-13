"""Scenario and scenario-grid configuration schemas."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from atbcr_analysis.config.models import ATBCRModelConfig


class ScenarioParametersConfig(BaseModel):
    """ATBCR parameters for one named YAML scenario."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epsilon: float = Field(ge=0)
    theta: float = Field(ge=0)
    mu: float = Field(default=0.1, gt=0, le=0.5)

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> ScenarioParametersConfig:
        if self.epsilon >= self.theta:
            raise ValueError("epsilon must be strictly smaller than theta")
        return self


class ScenarioConfig(ScenarioParametersConfig):
    """Named ATBCR experiment case."""

    name: str = Field(min_length=1)

    @computed_field
    @property
    def model(self) -> ATBCRModelConfig:
        """ATBCR model configuration derived from the scenario parameters."""

        return ATBCRModelConfig(epsilon=self.epsilon, theta=self.theta, mu=self.mu)


class DecimalRangeConfig(BaseModel):
    """Inclusive decimal range used by YAML scenario grids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: Decimal = Decimal("0")
    stop: Decimal = Decimal("1")
    step: Decimal = Decimal("0.05")

    @model_validator(mode="after")
    def _validate_step(self) -> DecimalRangeConfig:
        if self.step <= 0:
            raise ValueError("step must be positive")
        return self

    def values(self) -> tuple[Decimal, ...]:
        values: list[Decimal] = []
        current = self.start
        while current <= self.stop:
            values.append(current)
            current += self.step
        return tuple(values)


class ScenarioGridConfig(BaseModel):
    """YAML schema for generating ATBCR scenario grids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epsilon: DecimalRangeConfig = Field(default_factory=DecimalRangeConfig)
    theta: DecimalRangeConfig = Field(default_factory=DecimalRangeConfig)
    mu: Decimal = Decimal("0.1")
    epsilon_less_than_theta: bool = True

    def scenarios(self) -> tuple[ScenarioConfig, ...]:
        scenarios: list[ScenarioConfig] = []
        for epsilon in self.epsilon.values():
            for theta in self.theta.values():
                if self.epsilon_less_than_theta and epsilon >= theta:
                    continue
                name = f"e{decimal_name(epsilon)}_t{decimal_name(theta)}"
                scenarios.append(
                    ScenarioConfig(
                        name=name,
                        epsilon=float(epsilon),
                        theta=float(theta),
                        mu=float(self.mu),
                    ),
                )
        if not scenarios:
            raise ValueError("scenario_grid did not produce any valid scenarios")
        return tuple(scenarios)


def decimal_name(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text.replace("-", "m").replace(".", "p")
