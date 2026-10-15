"""CPU-only runner sampling controls and isolated launcher regressions."""

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np

from motionforge.models.hierarchical import (
    HierarchicalPolicyConfig,
    LocomotionModule,
    LocomotionNormalization,
    LocomotionObservation,
    StrategyModule,
    StrategyObservation,
)
from motionforge.rollout.hierarchical import (
    HierarchicalRolloutConfig,
    HierarchicalRolloutRunner,
    HierarchicalRolloutState,
)

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "training"))


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


launcher = load_script(
    "train_tag_motor_noise_ablation",
    SCRIPTS / "training/train_tag_motor_noise_ablation.py",
)
evaluator = load_script(
    "evaluate_hierarchical_tag_motor_sampling_test",
    SCRIPTS / "evaluation/evaluate_hierarchical_tag.py",
)


class SamplingTests(unittest.TestCase):
    def test_flag_resolution_and_metadata(self):
        for strategy in (False, True):
            for motor in (None, False, True):
                config = HierarchicalRolloutConfig(
                    stochastic_actions=strategy, stochastic_locomotion_actions=motor
                )
                resolved = strategy if motor is None else motor
                self.assertEqual(config.sample_locomotion_actions, resolved)
                metadata = evaluator._action_mode_metadata(config)
                self.assertEqual(
                    metadata["strategy_action_mode"],
                    "stochastic" if strategy else "deterministic",
                )
                self.assertEqual(
                    metadata["motor_action_mode"],
                    "stochastic" if resolved else "deterministic",
                )
                self.assertEqual(
                    metadata["action_mode"],
                    metadata["strategy_action_mode"]
                    if strategy == resolved
                    else "mixed",
                )
        self.assertIsNone(evaluator.Config().stochastic_locomotion_actions)

    def test_launcher_freeze_requirement_restore_and_no_default_change(self):
        with self.assertRaisesRegex(ValueError, "freeze_locomotion"):
            launcher.validate_motor_ablation(
                launcher.Config(
                    deterministic_frozen_locomotion=True, freeze_locomotion=False
                )
            )
        original = launcher.base.HierarchicalRolloutConfig
        for enabled in (False, True):
            seen = []

            def trainer(_config, seen=seen):
                seen.append(
                    launcher.base.HierarchicalRolloutConfig(stochastic_actions=True)
                )

            with patch.object(launcher.base, "main", trainer):
                launcher.main(
                    launcher.Config(
                        deterministic_frozen_locomotion=enabled, freeze_locomotion=True
                    )
                )
            self.assertIs(launcher.base.HierarchicalRolloutConfig, original)
            self.assertTrue(seen[0].stochastic_actions)
            self.assertEqual(seen[0].sample_locomotion_actions, not enabled)
            self.assertEqual(
                seen[0].stochastic_locomotion_actions, False if enabled else None
            )
        with (
            patch.object(launcher.base, "main", side_effect=RuntimeError("fixture")),
            self.assertRaises(RuntimeError),
        ):
            launcher.main(
                launcher.Config(
                    deterministic_frozen_locomotion=True, freeze_locomotion=True
                )
            )
        self.assertIs(launcher.base.HierarchicalRolloutConfig, original)

    def test_actual_jitted_runner_preserves_strategy_and_prng(self):
        policy_config = HierarchicalPolicyConfig(
            strategy_hidden_layer_sizes=(8,), locomotion_hidden_layer_sizes=(8,)
        )
        strategy_observation = StrategyObservation(
            opponent_position_local=jp.ones((2, 2)),
            opponent_velocity_local=jp.zeros((2, 2)),
            arena_center_local=jp.zeros((2, 2)),
            previous_command=jp.zeros((2, 3)),
            role=jp.eye(2),
        )
        locomotion_observation = LocomotionObservation(
            root_linear_velocity=jp.zeros((2, 3)),
            root_angular_velocity=jp.zeros((2, 3)),
            projected_gravity=jp.tile(jp.array([0.0, 0.0, -1.0]), (2, 1)),
            joint_position=jp.zeros((2, 29)),
            joint_velocity=jp.zeros((2, 29)),
            previous_action=jp.zeros((2, 29)),
            phase_cosine=jp.ones((2, 2)),
            phase_sine=jp.zeros((2, 2)),
        )
        strategy = StrategyModule((8,), policy_config.command_scale).init(
            jax.random.PRNGKey(1), strategy_observation.as_array()
        )["params"]
        motor = LocomotionModule((8,)).init(
            jax.random.PRNGKey(2), locomotion_observation.as_array(jp.zeros((2, 3)))
        )["params"]
        parameters = jax.tree.map(
            lambda value: jp.stack([value, value]),
            {"strategy": strategy, "locomotion": motor},
        )
        initial = HierarchicalRolloutState(
            environment=jp.zeros((2, 29)),
            strategy_actions=jp.zeros((2, 3)),
            commands=jp.zeros((2, 3)),
            previous_actions=jp.zeros((2, 29)),
            phases=jp.array([[0.0, jp.pi], [0.0, jp.pi]]),
            roles=jp.eye(2),
            strategy_values=jp.zeros(2),
            locomotion_values=jp.zeros(2),
            strategy_log_probabilities=jp.zeros(2),
            strategy_entropies=jp.zeros(2),
            locomotion_log_probabilities=jp.zeros(2),
            locomotion_entropies=jp.zeros(2),
            policy_rng=jax.random.PRNGKey(7),
            locomotion_steps=jp.asarray(0),
            strategy_updates=jp.asarray(0),
        )

        def run(config):
            runner = object.__new__(HierarchicalRolloutRunner)
            runner.config = config
            runner.policy = SimpleNamespace(config=policy_config)
            runner.observation_config = None
            runner.locomotion_layout = None
            runner.phase_delta = jp.asarray(2 * jp.pi * 0.02 * 1.375)
            runner.locomotion_normalization = LocomotionNormalization(
                jp.zeros(103), jp.ones(103)
            )
            runner.environment = SimpleNamespace(
                default_joint_targets=jp.zeros((2, 29)),
                step=lambda _state, targets: targets,
            )
            runner._agent_parameters = lambda _segment: parameters
            runner._project_boundary_safe_commands = lambda _state, commands: commands
            with (
                patch(
                    "motionforge.rollout.hierarchical.strategy_observations",
                    return_value=strategy_observation,
                ),
                patch(
                    "motionforge.rollout.hierarchical.locomotion_observations",
                    return_value=locomotion_observation,
                ),
            ):
                return jax.jit(
                    lambda state: runner.step(
                        SimpleNamespace(opponent_controller="learned"), state
                    )
                )(initial)

        for strategy_sampling in (False, True):
            config = HierarchicalRolloutConfig(stochastic_actions=strategy_sampling)
            default = run(config)
            explicit = run(
                replace(config, stochastic_locomotion_actions=strategy_sampling)
            )
            for a, b in zip(
                jax.tree.leaves(default), jax.tree.leaves(explicit), strict=True
            ):
                np.testing.assert_array_equal(a, b)
        stochastic = run(HierarchicalRolloutConfig(stochastic_actions=True))
        mixed = run(
            HierarchicalRolloutConfig(
                stochastic_actions=True, stochastic_locomotion_actions=False
            )
        )
        for field in (
            "strategy_actions",
            "commands",
            "strategy_values",
            "strategy_log_probabilities",
            "strategy_entropies",
            "policy_rng",
        ):
            np.testing.assert_array_equal(
                getattr(stochastic, field), getattr(mixed, field)
            )
        self.assertFalse(
            np.array_equal(stochastic.previous_actions, mixed.previous_actions)
        )
        np.testing.assert_array_equal(mixed.locomotion_log_probabilities, 0.0)


if __name__ == "__main__":
    unittest.main()
