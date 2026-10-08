"""Command line interface for ATBCR experiments."""

from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

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
)
from atbcr_analysis.experiments import ExperimentResult, run_monte_carlo
from atbcr_analysis.normalizers import NormalizationConfig
from atbcr_analysis.persistence import write_experiment_outputs


def main(argv: list[str] | None = None) -> int:
    config_path = _parse_config_path(argv)
    if config_path is not None:
        parser = _build_parser()
        overrides = _parse_cli_overrides(argv)
        overrides.pop("config", None)
        config = _read_yaml_config(config_path)
        for experiment in _iter_yaml_experiments(config):
            args = _args_from_mapping(parser, experiment, overrides)
            _run_experiment(args)
        return 0

    parser = _build_parser()
    args = parser.parse_args(argv)
    _run_experiment(args)
    return 0


def _run_experiment(args: argparse.Namespace) -> None:
    scenarios = _select_scenarios(args)
    variants = _select_variants(args)
    normalizations = _select_normalizations(args)
    output_dir = Path(args.output_dir)

    graph = _build_graph_config(args)
    metrics = MetricsConfig(
        extremized_threshold=args.extremized_threshold,
        cluster_tolerance=args.cluster_tolerance,
        record_every=args.record_every,
        store_opinion_snapshots=not args.no_snapshots,
    )
    opinion_initializer = _build_initializer_config(args)
    base_simulation = SimulationConfig(
        name=scenarios[0].name,
        model=scenarios[0].model,
        graph=graph,
        variant="baseline_01",
        domain="bounded_01",
        steps=args.steps,
        metrics=metrics,
        opinion_initializer=opinion_initializer,
        normalization=NormalizationConfig(),
    )
    monte_carlo = MonteCarloConfig(
        runs=args.runs,
        seed=args.seed,
        graph_seed=args.graph_seed,
        opinion_seed=args.opinion_seed,
        dynamics_seed=args.dynamics_seed,
        workers=args.workers,
    )

    results: list[ExperimentResult] = []
    for scenario in scenarios:
        for variant in variants:
            variant_base = base_simulation.with_variant(variant, scenario)
            for normalization in normalizations:
                simulation = _apply_normalization(variant_base, normalization)
                progress = None
                if not args.no_progress:
                    progress = _ProgressBar(_progress_label(simulation))
                try:
                    results.append(
                        run_monte_carlo(
                            simulation,
                            monte_carlo,
                            progress_callback=progress,
                        ),
                    )
                finally:
                    if progress is not None:
                        progress.finish()

    write_experiment_outputs(
        output_dir,
        results,
        {
            "args": vars(args),
            "scenarios": list(scenarios),
            "variants": list(variants),
            "normalizations": list(normalizations),
        },
    )
    _print_summary(results)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atbcr",
        description="Run Monte Carlo experiments for the ATBCR opinion dynamics model.",
    )
    parser.add_argument("--config", type=Path, help="YAML file with one or more experiment runs.")
    parser.add_argument(
        "--scenarios",
        default="custom:0.3:0.8:0.1",
        help="Comma-separated scenarios as name:epsilon:theta:mu.",
    )
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--theta", "--gamma", dest="theta", type=float, default=0.8)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--model", default="atbcr")
    parser.add_argument("--variants", default="baseline_01,bounded_m11,unbounded")
    parser.add_argument("--graph", default="complete")
    parser.add_argument("--agents", type=int, default=1000)
    parser.add_argument("--ba-m", type=int, default=3)
    parser.add_argument("--er-probability", type=float, default=0.02)
    parser.add_argument("--watts-k", type=int, default=6)
    parser.add_argument("--watts-rewire-probability", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=13_500)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--workers", type=_parse_workers, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--graph-seed", type=int)
    parser.add_argument("--opinion-seed", type=int)
    parser.add_argument("--dynamics-seed", type=int)
    parser.add_argument("--extremized-threshold", type=float, default=0.9)
    parser.add_argument("--cluster-tolerance", type=float, default=0.001)
    parser.add_argument("--initializer", choices=["uniform", "binary_concern"], default="uniform")
    parser.add_argument("--initial-low", type=float, default=0.0)
    parser.add_argument("--initial-high", type=float, default=1.0)
    parser.add_argument("--initial-concern-share", type=float, default=0.25)
    parser.add_argument("--concern-low", type=float, default=0.75)
    parser.add_argument("--concern-high", type=float, default=1.0)
    parser.add_argument("--non-concern-low", type=float, default=0.0)
    parser.add_argument("--non-concern-high", type=float, default=0.75)
    parser.add_argument(
        "--normalizer",
        choices=["none", "max_abs", "signed_log_max_abs"],
        default="none",
    )
    parser.add_argument("--normalization-every", default="")
    parser.add_argument("--record-every", type=int, default=500)
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-dir", default="runs/experiment")
    return parser


class _ProgressBar:
    def __init__(self, label: str, *, width: int = 28) -> None:
        self.label = label
        self.width = width
        self.completed = 0
        self.total = 0

    def __call__(self, completed: int, total: int) -> None:
        self.completed = completed
        self.total = total
        ratio = completed / total if total else 1.0
        filled = min(self.width, int(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        remaining = max(total - completed, 0)
        percent = int(ratio * 100)
        sys.stderr.write(
            f"\r{self.label} [{bar}] {completed}/{total} "
            f"({percent:3d}%) quedan {remaining}",
        )
        sys.stderr.flush()

    def finish(self) -> None:
        if self.total == 0:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()


def _progress_label(simulation: SimulationConfig) -> str:
    normalizer = simulation.normalization.kind
    if simulation.normalization.every is not None:
        normalizer = f"{normalizer}/{simulation.normalization.every}"
    return f"{simulation.name}/{simulation.variant}/{normalizer}"


def _parse_workers(value: Any) -> int:
    if isinstance(value, str) and value.lower() in {"auto", "max"}:
        return os.cpu_count() or 1
    try:
        workers = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "workers must be a positive integer, 'auto', or 'max'",
        ) from error
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be at least 1")
    return workers


def _parse_config_path(argv: list[str] | None) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    args, _ = parser.parse_known_args(argv)
    return args.config


def _parse_cli_overrides(argv: list[str] | None) -> dict[str, Any]:
    parser = _build_parser()
    for action in parser._actions:
        if action.option_strings:
            action.default = argparse.SUPPRESS
    return vars(parser.parse_args(argv))


def _read_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        raise ValueError(f"YAML config is empty: {path}")
    if not isinstance(payload, dict):
        raise ValueError("YAML config must be a mapping")
    return payload


def _iter_yaml_experiments(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    explicit_defaults = config.get("defaults", {})
    if not isinstance(explicit_defaults, dict):
        raise ValueError("YAML 'defaults' must be a mapping")

    experiments = config.get("experiments")
    if experiments is None:
        return (_normalize_config_mapping(config, reserved_keys={"defaults"}),)
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("YAML 'experiments' must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    inline_defaults = {
        key: value
        for key, value in config.items()
        if key not in {"defaults", "experiments"}
    }
    defaults = {**inline_defaults, **explicit_defaults}
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each YAML experiment must be a mapping")
        merged = {**defaults, **experiment}
        normalized.append(_normalize_config_mapping(merged))
    return tuple(normalized)


def _normalize_config_mapping(
    config: dict[str, Any],
    *,
    reserved_keys: set[str] | None = None,
) -> dict[str, Any]:
    reserved = {"config", "experiments", *(reserved_keys or set())}
    normalized = {key.replace("-", "_"): value for key, value in config.items()}
    if "scenario_grid" in normalized:
        if "scenarios" in normalized:
            raise ValueError("Use either 'scenarios' or 'scenario_grid', not both")
        normalized["scenarios"] = _expand_scenario_grid(normalized.pop("scenario_grid"))
    parser = _build_parser()
    allowed = {
        action.dest
        for action in parser._actions
        if action.dest != argparse.SUPPRESS and action.dest not in reserved
    }
    unknown = set(normalized) - allowed - reserved
    if unknown:
        raise ValueError(f"Unsupported YAML config keys: {sorted(unknown)}")
    return {
        key: _normalize_config_value(key, value)
        for key, value in normalized.items()
        if key not in reserved
    }


def _normalize_config_value(key: str, value: Any) -> Any:
    if key in {"scenarios", "variants", "normalization_every"} and isinstance(value, list):
        return ",".join(str(item) for item in value)
    if key == "workers":
        return _parse_workers(value)
    return value


def _expand_scenario_grid(grid: Any) -> str:
    if not isinstance(grid, dict):
        raise ValueError("YAML 'scenario_grid' must be a mapping")
    epsilon = _decimal_range(grid.get("epsilon", {}), label="epsilon")
    theta = _decimal_range(grid.get("theta", {}), label="theta")
    mu = Decimal(str(grid.get("mu", "0.1")))
    strict_epsilon_less_than_theta = bool(grid.get("epsilon_less_than_theta", True))
    scenarios = []
    for epsilon_value in epsilon:
        for theta_value in theta:
            if strict_epsilon_less_than_theta and epsilon_value >= theta_value:
                continue
            name = (
                f"e{_decimal_name(epsilon_value)}"
                f"_t{_decimal_name(theta_value)}"
            )
            scenarios.append(f"{name}:{epsilon_value}:{theta_value}:{mu}")
    if not scenarios:
        raise ValueError("scenario_grid did not produce any valid scenarios")
    return ",".join(scenarios)


def _decimal_range(config: Any, *, label: str) -> list[Decimal]:
    if not isinstance(config, dict):
        raise ValueError(f"YAML scenario_grid.{label} must be a mapping")
    start = Decimal(str(config.get("start", 0)))
    stop = Decimal(str(config.get("stop", 1)))
    step = Decimal(str(config.get("step", "0.05")))
    if step <= 0:
        raise ValueError(f"YAML scenario_grid.{label}.step must be positive")
    values: list[Decimal] = []
    current = start
    while current <= stop:
        values.append(current)
        current += step
    return values


def _decimal_name(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text.replace("-", "m").replace(".", "p")


def _args_from_mapping(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    overrides: dict[str, Any],
) -> argparse.Namespace:
    values = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
    }
    values.update(config)
    values.update(overrides)
    return argparse.Namespace(**values)


def _build_graph_config(args: argparse.Namespace) -> GraphConfig:
    normalized = args.graph.lower().replace("-", "_")
    if normalized in {"complete", "fully_connected"}:
        return CompleteGraphConfig(n_agents=args.agents)
    if normalized in {"ring", "cycle"}:
        return RingGraphConfig(n_agents=args.agents)
    if normalized in {"barabasi_albert", "ba", "scale_free"}:
        return BarabasiAlbertGraphConfig(n_agents=args.agents, ba_m=args.ba_m)
    if normalized in {"erdos_renyi", "er"}:
        return ErdosRenyiGraphConfig(n_agents=args.agents, er_probability=args.er_probability)
    if normalized in {"watts_strogatz", "ws", "small_world"}:
        return WattsStrogatzGraphConfig(
            n_agents=args.agents,
            watts_k=args.watts_k,
            watts_rewire_probability=args.watts_rewire_probability,
        )
    raise ValueError(f"Unsupported graph kind: {args.graph}")


def _select_scenarios(args: argparse.Namespace) -> tuple[ScenarioConfig, ...]:
    if args.scenarios:
        return tuple(
            _parse_scenario(raw, args.model)
            for raw in args.scenarios.split(",")
            if raw.strip()
        )
    return (
        ScenarioConfig(
            name="custom",
            model=_build_model_config(args.model, args.epsilon, args.theta, args.mu),
        ),
    )


def _parse_scenario(raw: str, model_name: str) -> ScenarioConfig:
    parts = [part.strip() for part in raw.split(":")]
    if model_name == "atbcr" and len(parts) != 4:
        raise ValueError("scenarios must use name:epsilon:theta:mu")
    if model_name != "atbcr" and len(parts) != 1:
        raise ValueError("non-ATBCR scenarios must use name")
    if model_name != "atbcr":
        return ScenarioConfig(name=parts[0], model=ModelConfig(name=model_name))
    name, epsilon, theta, mu = parts
    return ScenarioConfig(
        name=name,
        model=ATBCRModelConfig(epsilon=float(epsilon), theta=float(theta), mu=float(mu)),
    )


def _build_model_config(model_name: str, epsilon: float, theta: float, mu: float) -> ModelConfig:
    if model_name == "atbcr":
        return ATBCRModelConfig(epsilon=epsilon, theta=theta, mu=mu)
    return ModelConfig(name=model_name)


def _select_variants(args: argparse.Namespace) -> tuple[VariantConfig, ...]:
    requested = {name.strip() for name in args.variants.split(",") if name.strip()}
    variants_by_name = {variant.name: variant for variant in DEFAULT_VARIANTS}
    missing = requested - set(variants_by_name)
    if missing:
        raise ValueError(f"Unsupported variants: {sorted(missing)}")

    selected = tuple(variants_by_name[name] for name in variants_by_name if name in requested)
    custom_initializer_requested = (
        args.initializer != "uniform"
        or args.initial_low != 0.0
        or args.initial_high != 1.0
        or args.initial_concern_share != 0.25
        or args.concern_low != 0.75
        or args.concern_high != 1.0
        or args.non_concern_low != 0.0
        or args.non_concern_high != 0.75
    )
    if custom_initializer_requested:
        custom_initializer = _build_initializer_config(args)
        selected = tuple(
            VariantConfig(
                name=variant.name,
                domain=variant.domain,
                threshold_scale=variant.threshold_scale,
                initializer=custom_initializer,
                normalizations=variant.normalizations,
            )
            for variant in selected
        )
    return selected


def _build_initializer_config(
    args: argparse.Namespace,
) -> UniformOpinionInitializerConfig | BinaryConcernOpinionInitializerConfig:
    if args.initializer == "uniform":
        return UniformOpinionInitializerConfig(low=args.initial_low, high=args.initial_high)
    if args.initializer == "binary_concern":
        return BinaryConcernOpinionInitializerConfig(
            concern_share=args.initial_concern_share,
            concern_low=args.concern_low,
            concern_high=args.concern_high,
            non_concern_low=args.non_concern_low,
            non_concern_high=args.non_concern_high,
        )
    raise ValueError(f"Unsupported opinion initializer: {args.initializer}")


def _select_normalizations(args: argparse.Namespace) -> tuple[NormalizationConfig, ...]:
    if args.normalizer == "none":
        return (NormalizationConfig(kind="none"),)

    frequencies = [
        int(raw)
        for raw in args.normalization_every.split(",")
        if raw.strip()
    ]
    if not frequencies:
        raise ValueError("--normalization-every is required when --normalizer is not none")
    return tuple(
        NormalizationConfig(kind=args.normalizer, every=frequency)
        for frequency in frequencies
    )


def _apply_normalization(
    config: SimulationConfig,
    normalization: NormalizationConfig,
) -> SimulationConfig:
    return SimulationConfig(
        name=config.name,
        model=config.model,
        graph=config.graph,
        variant=config.variant,
        domain=config.domain,
        steps=config.steps,
        metrics=config.metrics,
        opinion_initializer=config.opinion_initializer,
        normalization=normalization,
    )


def _print_summary(results: list[ExperimentResult]) -> None:
    for result in results:
        print(
            f"{result.scenario_name}/{result.variant_name}/"
            f"{result.normalizer_name}/{result.normalization_every}: "
            f"confidence={result.mean_confidence_frequency:.4f}, "
            f"inaction={result.mean_inaction_frequency:.4f}, "
            f"repulsion={result.mean_repulsion_frequency:.4f}, "
            f"max_abs={result.mean_max_abs_opinion:.4f}",
        )


if __name__ == "__main__":
    raise SystemExit(main())
