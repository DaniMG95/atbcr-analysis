from __future__ import annotations

import math
import random
import unittest

from pydantic import ValidationError

from atbcr_analysis.config import (
    ATBCRModelConfig,
    ATBCRShiftModelConfig,
    CompleteGraphConfig,
    FileConfig,
    MetricsConfig,
    SimulationConfig,
    UniformOpinionInitializerConfig,
)
from atbcr_analysis.graphs import complete_graph_edges
from atbcr_analysis.models import ATBCRShiftModel, ModelFactory
from atbcr_analysis.simulation import simulate_atbcr


class ATBCRShiftTests(unittest.TestCase):
    def _config(self, model: ATBCRModelConfig, *, steps: int = 40) -> SimulationConfig:
        return SimulationConfig(
            name="shift",
            model=model,
            graph=CompleteGraphConfig(n_agents=12),
            variant="bounded_m11",
            domain="bounded_m11",
            steps=steps,
            metrics=MetricsConfig(record_every=1),
            opinion_initializer=UniformOpinionInitializerConfig(low=-1.0, high=1.0),
        )

    def test_both_models_are_registered_and_legacy_config_still_parses(self) -> None:
        self.assertIn("atbcr", ModelFactory.available_models())
        self.assertIn("atbcr_shift", ModelFactory.available_models())
        legacy = FileConfig.from_mapping(
            {"experiments": [{"scenarios": {"legacy": {"epsilon": 0.2, "theta": 0.7}}}]},
        )
        shift = FileConfig.from_mapping(
            {
                "experiments": [
                    {
                        "scenarios": {
                            "new": {
                                "model": "atbcr_shift",
                                "epsilon": 0.2,
                                "theta": 0.7,
                                "p": 0.9,
                                "shift_lambda": 0.01,
                            },
                        },
                    },
                ],
            },
        )
        self.assertEqual(legacy.runs[0].resolved_scenarios[0].model.name, "atbcr")
        self.assertEqual(shift.runs[0].resolved_scenarios[0].model.name, "atbcr_shift")

    def test_p_one_matches_bounded_atbcr_trajectory_without_shift_steps(self) -> None:
        original = self._config(ATBCRModelConfig(epsilon=0.4, theta=1.4, mu=0.2))
        shifted = self._config(
            ATBCRShiftModelConfig(
                epsilon=0.4,
                theta=1.4,
                mu=0.2,
                p=1.0,
                shift_lambda=0.6,
            ),
        )
        original_result = simulate_atbcr(original, seed=123)
        shifted_result = simulate_atbcr(shifted, seed=123)

        self.assertEqual(
            [snapshot.opinions for snapshot in original_result.trajectory],
            [snapshot.opinions for snapshot in shifted_result.trajectory],
        )
        self.assertEqual(shifted_result.final_metrics.shift_steps, 0)
        self.assertEqual(shifted_result.final_metrics.atbcr_steps, shifted.steps)

    def test_zero_shift_keeps_opinions_and_counts_every_step(self) -> None:
        config = self._config(
            ATBCRShiftModelConfig(p=0.0, shift_lambda=0.0),
            steps=25,
        )
        result = simulate_atbcr(config, seed=8)

        self.assertEqual(result.trajectory[0].opinions, result.trajectory[-1].opinions)
        self.assertEqual(result.final_metrics.shift_steps, 25)
        self.assertEqual(result.final_metrics.shift_frequency, 1.0)
        self.assertEqual(result.final_metrics.atbcr_steps, 0)
        self.assertTrue(math.isnan(result.final_metrics.confidence_frequency))

    def test_signed_shifts_change_only_selected_agent_and_clip_both_ends(self) -> None:
        edges = complete_graph_edges(3)
        for shift_lambda, opinions, expected in (
            (0.4, [-0.5, 0.9, 0.0], [-0.5, 1.0, 0.0]),
            (-0.4, [0.5, -0.9, 0.0], [0.5, -1.0, 0.0]),
        ):
            model = ATBCRShiftModel(ATBCRShiftModelConfig(p=0.0, shift_lambda=shift_lambda))
            model.step(opinions, ((), (), ()), edges, _AgentOneRandom())
            self.assertEqual(opinions, expected)

    def test_mixed_execution_is_reproducible_and_partitions_steps(self) -> None:
        config = self._config(ATBCRShiftModelConfig(p=0.55, shift_lambda=0.03), steps=100)
        first = simulate_atbcr(config, seed=99)
        second = simulate_atbcr(config, seed=99)

        self.assertEqual(first.final_opinions, second.final_opinions)
        self.assertEqual(first.final_metrics, second.final_metrics)
        self.assertEqual(
            first.final_metrics.shift_steps + first.final_metrics.atbcr_steps,
            config.steps,
        )

    def test_new_parameter_ranges_are_validated(self) -> None:
        for values in ({"p": -0.1}, {"p": 1.1}, {"shift_lambda": -1.1}, {"shift_lambda": 1.1}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                ATBCRShiftModelConfig(**values)


class _AgentOneRandom(random.Random):
    def randrange(self, *args: object, **kwargs: object) -> int:
        return 1


if __name__ == "__main__":
    unittest.main()
