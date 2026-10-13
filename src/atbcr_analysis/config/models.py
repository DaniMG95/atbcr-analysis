"""Model configuration schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


def scale_model_thresholds(model: ModelConfigType, factor: float) -> ModelConfigType:
    if isinstance(model, ATBCRModelConfig):
        return model.scale_thresholds(factor)
    if factor != 1.0:
        raise ValueError("threshold scaling is only supported by ATBCRModelConfig")
    return model


def build_model_config(
    model_name: str,
    epsilon: object,
    theta: object,
    mu: object,
) -> ModelConfigType:
    if model_name == "atbcr":
        return ATBCRModelConfig(epsilon=float(epsilon), theta=float(theta), mu=float(mu))
    return ModelConfig(name=model_name)
