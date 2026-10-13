"""Graph configuration schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
