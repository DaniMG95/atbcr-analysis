"""Command line interface for ATBCR experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser = _build_parser()
    args = parser.parse_args(argv)
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
    monte_carlo = MonteCarloConfig(runs=args.runs, seed=args.seed, workers=args.workers)

    results: list[ExperimentResult] = []
    for scenario in scenarios:
        for variant in variants:
            variant_base = base_simulation.with_variant(variant, scenario)
            for normalization in normalizations:
                simulation = _apply_normalization(variant_base, normalization)
                results.append(run_monte_carlo(simulation, monte_carlo))

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
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atbcr",
        description="Run Monte Carlo experiments for the ATBCR opinion dynamics model.",
    )
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
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
    parser.add_argument("--normalizer", choices=["none", "max_abs", "signed_log_max_abs"], default="none")
    parser.add_argument("--normalization-every", default="")
    parser.add_argument("--record-every", type=int, default=500)
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--output-dir", default="runs/experiment")
    return parser


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
        return tuple(_parse_scenario(raw, args.model) for raw in args.scenarios.split(",") if raw.strip())
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
    return tuple(NormalizationConfig(kind=args.normalizer, every=frequency) for frequency in frequencies)


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
