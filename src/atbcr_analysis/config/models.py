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
    epsilon: float = Field(default=0.3, ge=0, allow_inf_nan=False)
    theta: float = Field(default=0.8, ge=0, allow_inf_nan=False)
    mu: float = Field(default=0.25, gt=0, le=0.5, allow_inf_nan=False)
    threshold_mode: Literal["fixed", "max_abs"] = "fixed"
    threshold_update_every: int = Field(default=1, ge=1)

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
            threshold_mode=self.threshold_mode,
            threshold_update_every=self.threshold_update_every,
        )

    def with_threshold_policy(
        self,
        mode: Literal["fixed", "max_abs"],
        update_every: int,
    ) -> ATBCRModelConfig:
        """Return the same ATBCR parameters with a threshold update policy."""

        return self.model_copy(
            update={
                "threshold_mode": mode,
                "threshold_update_every": update_every,
            },
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
