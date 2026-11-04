from __future__ import annotations

import math
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from atbcr_analysis.config import (
    ATBCRModelConfig,
    FileConfig,
    ScenarioConfig,
    SimulationConfig,
    VariantConfig,
)
from atbcr_analysis.models.atbcr import ATBCRModel


class DynamicThresholdTests(unittest.TestCase):
    def test_thresholds_scale_relative_to_initial_maximum(self) -> None:
        model = ATBCRModel(
            ATBCRModelConfig(
                epsilon=0.4,
                theta=1.4,
                mu=0.1,
                threshold_mode="max_abs",
            ),
            domain="unbounded",
        )
        opinions = [0.0, 1.0, 2.0]
        model.initialize(opinions)

        model._thresholds(opinions)
        opinions[:] = [0.0, 1.0, 4.0]
        thresholds = model._thresholds(opinions)

        self.assertEqual(model.initial_max_abs, 2.0)
        self.assertAlmostEqual(thresholds[0], 0.8)
        self.assertAlmostEqual(thresholds[1], 2.8)

    def test_thresholds_only_refresh_after_configured_number_of_steps(self) -> None:
        model = ATBCRModel(
            ATBCRModelConfig(
                epsilon=0.4,
                theta=1.4,
                mu=0.1,
                threshold_mode="max_abs",
                threshold_update_every=2,
            ),
            domain="unbounded",
        )
        opinions = [0.0, 0.0, 1.0]
        model.initialize(opinions)

        model._update_pair(opinions, 0, 1)
        opinions[:] = [0.0, 0.0, 3.0]
        model._update_pair(opinions, 0, 1)
        self.assertEqual(model.effective_thresholds, (0.4, 1.4))
        opinions[:] = [0.0, 0.0, 4.0]
        model._update_pair(opinions, 0, 1)

        self.assertAlmostEqual(model.effective_thresholds[0], 1.6)
        self.assertAlmostEqual(model.effective_thresholds[1], 5.6)

    def test_zero_initial_maximum_keeps_initial_thresholds(self) -> None:
        model = ATBCRModel(
            ATBCRModelConfig(
                epsilon=0.4,
                theta=1.4,
                mu=0.1,
                threshold_mode="max_abs",
            ),
            domain="unbounded",
        )
        opinions = [0.0, 0.0]
        model.initialize(opinions)

        model._thresholds(opinions)
        opinions[:] = [3.0, -3.0]
        thresholds = model._thresholds(opinions)

        self.assertEqual(model.initial_max_abs, 0.0)
        self.assertEqual(thresholds, (0.4, 1.4))

    def test_non_finite_threshold_configuration_is_rejected(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ATBCRModelConfig(epsilon=value, theta=1.4, mu=0.1)

    def test_dynamic_threshold_overflow_raises_explicit_error(self) -> None:
        model = ATBCRModel(
            ATBCRModelConfig(
                epsilon=0.4,
                theta=1.4,
                mu=0.1,
                threshold_mode="max_abs",
            ),
            domain="unbounded",
        )
        opinions = [1e-308, 0.0]
        model.initialize(opinions)
        model._thresholds(opinions)
        opinions[:] = [1e308, 0.0]

        with self.assertRaisesRegex(FloatingPointError, "dynamic threshold overflow"):
            model._thresholds(opinions)

    def test_opinion_overflow_raises_explicit_error(self) -> None:
        model = ATBCRModel(
            ATBCRModelConfig(
                epsilon=0.4,
                theta=1.4,
                mu=0.1,
                threshold_mode="max_abs",
                threshold_update_every=100,
            ),
            domain="unbounded",
        )
        opinions = [1e308, -1e308]
        model.initialize(opinions)

        with self.assertRaisesRegex(FloatingPointError, "opinion overflow"):
            model._update_pair(opinions, 0, 1)

    def test_variant_applies_dynamic_policy_after_static_domain_scale(self) -> None:
        scenario = ScenarioConfig(name="repulsive", epsilon=0.2, theta=0.7, mu=0.1)
        variant = VariantConfig(
            name="dynamic",
            domain="unbounded",
            threshold_scale=2.0,
            threshold_mode="max_abs",
            threshold_update_every=100,
        )

        config = SimulationConfig(name="base", model=scenario.model).with_variant(
            variant,
            scenario,
        )

        self.assertEqual(config.model.epsilon, 0.4)
        self.assertEqual(config.model.theta, 1.4)
        self.assertEqual(config.model.threshold_mode, "max_abs")
        self.assertEqual(config.model.threshold_update_every, 100)

    def test_experiment_07_has_no_normalization_or_clipping(self) -> None:
        payload = yaml.safe_load(
            Path("experiments/experiment_0.7.yaml").read_text(encoding="utf-8"),
        )
        run = FileConfig.from_mapping(payload).runs[0]

        self.assertEqual(len(run.resolved_variants), 5)
        self.assertEqual(run.normalizations[0].kind, "none")
        for variant in run.resolved_variants:
            self.assertEqual(variant.domain, "unbounded")
            self.assertEqual(variant.threshold_scale, 2.0)
            self.assertEqual(variant.threshold_mode, "max_abs")
            self.assertEqual((variant.initializer.low, variant.initializer.high), (-1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
