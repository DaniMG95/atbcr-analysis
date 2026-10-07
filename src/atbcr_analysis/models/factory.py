"""Factory and registry for opinion dynamics models."""

from __future__ import annotations

from collections.abc import Callable

from atbcr_analysis.config import ATBCRModelConfig, ModelConfig
from atbcr_analysis.models.atbcr import ATBCRModel
from atbcr_analysis.models.base import OpinionDynamicsModel

ModelBuilder = Callable[[ModelConfig, str], OpinionDynamicsModel]


class ModelFactory:
    """Registry-backed factory for opinion dynamics models."""

    _builders: dict[str, ModelBuilder] = {}

    @classmethod
    def register(cls, name: str, builder: ModelBuilder) -> None:
        normalized = _normalize_name(name)
        if not normalized:
            raise ValueError("model name cannot be empty")
        cls._builders[normalized] = builder

    @classmethod
    def create(
        cls,
        model_config: ModelConfig,
        *,
        domain: str,
    ) -> OpinionDynamicsModel:
        normalized = _normalize_name(model_config.name)
        if normalized not in cls._builders:
            raise ValueError(f"Unsupported model: {model_config.name}")
        return cls._builders[normalized](model_config, domain)

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._builders))


def register_model(name: str, builder: ModelBuilder) -> None:
    """Register a model builder in the package-level factory."""

    ModelFactory.register(name, builder)


def build_model(model_config: ModelConfig, *, domain: str) -> OpinionDynamicsModel:
    """Build a model from config."""

    return ModelFactory.create(model_config, domain=domain)


def _build_atbcr_model(
    model_config: ModelConfig,
    domain: str,
) -> OpinionDynamicsModel:
    if not isinstance(model_config, ATBCRModelConfig):
        raise TypeError("ATBCR requires ATBCRModelConfig")
    return ATBCRModel(config=model_config, domain=domain)


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "_")


ModelFactory.register("atbcr", _build_atbcr_model)
