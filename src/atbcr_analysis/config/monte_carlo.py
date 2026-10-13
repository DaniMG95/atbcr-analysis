"""Monte Carlo execution configuration schemas."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MonteCarloConfig(BaseModel):
    """Monte Carlo execution settings."""

    model_config = ConfigDict(frozen=True)

    runs: int = Field(default=20, ge=1)
    seed: int = 7
    graph_seed: int | None = None
    opinion_seed: int | None = None
    dynamics_seed: int | None = None
    workers: int = Field(default=1, ge=1)

    @field_validator("workers", mode="before")
    @classmethod
    def _parse_workers(cls, value: Any) -> Any:
        if isinstance(value, str) and value.lower() in {"auto", "max"}:
            return os.cpu_count() or 1
        return value
