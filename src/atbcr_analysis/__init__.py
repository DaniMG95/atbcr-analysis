"""Tools for ATBCR dynamic opinion Monte Carlo experiments."""

from atbcr_analysis.config import (
    ATBCRModelConfig,
    DEFAULT_VARIANTS,
    BarabasiAlbertGraphConfig,
    BinaryConcernOpinionInitializerConfig,
    CompleteGraphConfig,
    ErdosRenyiGraphConfig,
    GraphConfig,
    MetricsConfig,
    ModelConfig,
    MonteCarloConfig,
    RingGraphConfig,
    ScenarioConfig,
    SimulationConfig,
    UniformOpinionInitializerConfig,
    VariantConfig,
    WattsStrogatzGraphConfig,
    OpinionInitializerConfig,
)
from atbcr_analysis.experiment_grid import build_paired_grid
from atbcr_analysis.experiments import ExperimentResult, run_monte_carlo
from atbcr_analysis.graphs import GraphFactory, register_graph
from atbcr_analysis.initializers import (
    BinaryConcernOpinionInitializer,
    UniformOpinionInitializer,
)
from atbcr_analysis.normalizers import (
    MaxAbsNormalizer,
    NoNormalizer,
    NormalizationConfig,
    SignedLogMaxAbsNormalizer,
)
from atbcr_analysis.models import ATBCRModel, ModelFactory, OpinionDynamicsModel, register_model
from atbcr_analysis.simulation import SimulationResult, SimulationRunner, simulate_atbcr

__all__ = [
    "ATBCRModel",
    "ATBCRModelConfig",
    "BarabasiAlbertGraphConfig",
    "BinaryConcernOpinionInitializer",
    "BinaryConcernOpinionInitializerConfig",
    "CompleteGraphConfig",
    "DEFAULT_VARIANTS",
    "ErdosRenyiGraphConfig",
    "ExperimentResult",
    "GraphFactory",
    "GraphConfig",
    "MetricsConfig",
    "ModelConfig",
    "ModelFactory",
    "MonteCarloConfig",
    "MaxAbsNormalizer",
    "NoNormalizer",
    "NormalizationConfig",
    "OpinionDynamicsModel",
    "OpinionInitializerConfig",
    "RingGraphConfig",
    "ScenarioConfig",
    "SimulationConfig",
    "SimulationResult",
    "SimulationRunner",
    "SignedLogMaxAbsNormalizer",
    "UniformOpinionInitializer",
    "UniformOpinionInitializerConfig",
    "VariantConfig",
    "WattsStrogatzGraphConfig",
    "build_paired_grid",
    "register_graph",
    "register_model",
    "run_monte_carlo",
    "simulate_atbcr",
]
