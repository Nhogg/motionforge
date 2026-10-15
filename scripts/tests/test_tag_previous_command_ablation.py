"""CPU policy, gradient, Adam and checkpoint tests for the learner constraint."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
from flax.core import freeze, unfreeze

from motionforge.models import tanh_normal_log_prob
from motionforge.models.hierarchical import (
    HierarchicalPolicyConfig,
    LocomotionModule,
    StrategyModule,
)
from motionforge.rollout import AdvantageTargets, LocomotionBatch, StrategyBatch
from motionforge.training.hierarchical_checkpoint import (
    load_hierarchical_checkpoint,
    save_hierarchical_checkpoint,
)
from motionforge.training.hierarchical_ppo import locomotion_ppo_loss, strategy_ppo_loss
from motionforge.training.hierarchical_update import (
    HierarchicalPpoUpdater,
    HierarchicalUpdateConfig,
)
from motionforge.training.tag_previous_command_ablation import (
    PreviousCommandConstrainedUpdater,
    constrain_hierarchical_previous_command,
)

TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING))
SPEC = importlib.util.spec_from_file_location(
    "train_tag_previous_command_ablation",
    TRAINING / "train_tag_previous_command_ablation.py",
)
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def equal_trees(test, left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


class PreviousCommandTests(unittest.TestCase):
    def setUp(self):
        self.policy_config = HierarchicalPolicyConfig(
            strategy_hidden_layer_sizes=(8, 8), locomotion_hidden_layer_sizes=(8, 8)
        )
        self.strategy = StrategyModule((8, 8), self.policy_config.command_scale)
        self.motor = LocomotionModule((8, 8))
        self.observation = jax.random.normal(jax.random.PRNGKey(1), (8, 11)) * 0.2
        self.observation = self.observation.at[:, -2:].set(jp.tile(jp.eye(2), (4, 1)))
        self.motor_observation = (
            jax.random.normal(jax.random.PRNGKey(2), (8, 103)) * 0.1
        )
        self.original = freeze(
            {
                "strategy": self.strategy.init(jax.random.PRNGKey(3), self.observation)[
                    "params"
                ],
                "locomotion": self.motor.init(
                    jax.random.PRNGKey(4), self.motor_observation
                )["params"],
            }
        )
        self.parameters = constrain_hierarchical_previous_command(self.original)
        output = self.strategy.apply(
            {"params": self.parameters["strategy"]}, self.observation
        )
        motor_output = self.motor.apply(
            {"params": self.parameters["locomotion"]}, self.motor_observation
        )
        command = jp.clip(
            output.command + jax.random.normal(jax.random.PRNGKey(5), (8, 3)) * 0.03,
            -jp.array([0.99, 0.49, 0.99]),
            jp.array([0.99, 0.49, 0.99]),
        )
        zeros = jp.zeros(8)
        valid = jp.array([True, True, False, True, True, True, False, True])
        self.strategy_batch = StrategyBatch(
            observation=self.observation,
            command=command,
            value=output.value,
            log_probability=tanh_normal_log_prob(
                command,
                output.location,
                output.log_std,
                jp.array(self.policy_config.command_scale),
            ),
            entropy=zeros,
            reward=zeros,
            done=jp.zeros(8, bool),
            valid=valid,
            role=self.observation[:, -2:],
            episode_id=jp.arange(8),
            environment_id=jp.arange(8),
        )
        self.motor_batch = LocomotionBatch(
            observation=self.motor_observation,
            action=motor_output.action,
            value=motor_output.value,
            log_probability=tanh_normal_log_prob(
                motor_output.action,
                motor_output.location,
                motor_output.log_std,
                jp.ones(29),
            ),
            entropy=zeros,
            reward=zeros,
            done=jp.zeros(8, bool),
            valid=valid,
            command_tracking_error=jp.zeros((8, 3)),
            episode_id=jp.arange(8),
            environment_id=jp.arange(8),
        )
        self.strategy_targets = AdvantageTargets(
            jp.linspace(-1, 1, 8), output.value + 0.2
        )
        self.motor_targets = AdvantageTargets(
            jp.linspace(1, -1, 8), motor_output.value - 0.2
        )
        self.config = HierarchicalUpdateConfig(freeze_locomotion=True, minibatch_size=4)

    def test_projection_changes_only_learner_rows_and_not_historical_source(self):
        expected = unfreeze(self.original)
        expected["strategy"]["encoder"]["Dense_0"]["kernel"] = (
            expected["strategy"]["encoder"]["Dense_0"]["kernel"].at[6:9].set(0)
        )
        equal_trees(self, self.parameters, freeze(expected))
        self.assertTrue(
            np.any(
                np.asarray(self.original["strategy"]["encoder"]["Dense_0"]["kernel"])[
                    6:9
                ]
                != 0
            )
        )
        for name in ("pursuer_head", "evader_head"):
            equal_trees(
                self, self.parameters["strategy"][name], self.original["strategy"][name]
            )
        equal_trees(self, self.parameters["locomotion"], self.original["locomotion"])

    def test_jitted_policy_ignores_previous_command_but_uses_target_bearing(self):
        actor = jax.jit(
            lambda parameters, observation: self.strategy.apply(
                {"params": parameters}, observation
            )
        )
        baseline = actor(self.parameters["strategy"], self.observation)
        previous_changed = self.observation.at[:, 6:9].set(jp.array([1.0, -0.8, 0.6]))
        equal_trees(
            self, baseline, actor(self.parameters["strategy"], previous_changed)
        )
        bearing_changed = self.observation.at[:, :2].set(jp.array([-0.8, 0.9]))
        self.assertGreater(
            float(
                jp.max(
                    jp.abs(
                        baseline.command
                        - actor(self.parameters["strategy"], bearing_changed).command
                    )
                )
            ),
            1e-5,
        )
        historical_before = actor(self.original["strategy"], self.observation)
        historical_after = actor(self.original["strategy"], previous_changed)
        self.assertGreater(
            float(jp.max(jp.abs(historical_before.command - historical_after.command))),
            1e-5,
        )

    def test_exact_strategy_interception_zero_gradient_and_motor_loss_unchanged(self):
        constrained = PreviousCommandConstrainedUpdater(
            self.config, policy_config=self.policy_config
        )
        original = HierarchicalPpoUpdater(self.config, policy_config=self.policy_config)
        arguments = (
            strategy_ppo_loss,
            self.parameters["strategy"],
            self.strategy_batch,
            self.strategy_targets,
            constrained.strategy_loss_config,
        )
        metrics, gradients = jax.jit(lambda: constrained._loss_and_grad(*arguments))()
        expected_metrics, expected_gradients = original._loss_and_grad(*arguments)
        # Compiled and eager float32 reductions can differ by rounding.
        for actual, expected_value in zip(
            jax.tree.leaves(metrics), jax.tree.leaves(expected_metrics), strict=True
        ):
            np.testing.assert_allclose(actual, expected_value, rtol=1e-4, atol=1e-6)
        expected = unfreeze(expected_gradients)
        expected["encoder"]["Dense_0"]["kernel"] = (
            expected["encoder"]["Dense_0"]["kernel"].at[6:9].set(0.0)
        )
        for actual, expected_value in zip(
            jax.tree.leaves(gradients), jax.tree.leaves(freeze(expected)), strict=True
        ):
            np.testing.assert_allclose(actual, expected_value, rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(
            gradients["encoder"]["Dense_0"]["kernel"][6:9], 0.0
        )
        self.assertGreater(
            float(
                jp.max(jp.abs(expected_gradients["encoder"]["Dense_0"]["kernel"][6:9]))
            ),
            1e-7,
        )
        arguments = (
            locomotion_ppo_loss,
            self.parameters["locomotion"],
            self.motor_batch,
            self.motor_targets,
            constrained.locomotion_loss_config,
        )
        equal_trees(
            self,
            constrained._loss_and_grad(*arguments),
            original._loss_and_grad(*arguments),
        )
        self.assertIs(
            constrained._constrained_loss(locomotion_ppo_loss), locomotion_ppo_loss
        )

    def test_real_repeated_jitted_updates_keep_parameters_and_adam_rows_zero(self):
        updater = PreviousCommandConstrainedUpdater(
            self.config, policy_config=self.policy_config
        )
        initial = updater.initialize(self.parameters)
        state = initial
        update = jax.jit(updater.update)
        for _ in range(5):
            state, metrics = update(
                state,
                self.strategy_batch,
                self.strategy_targets,
                self.motor_batch,
                self.motor_targets,
            )
            self.assertTrue(bool(metrics.strategy_updated))
            self.assertFalse(bool(metrics.locomotion_updated))
            np.testing.assert_array_equal(
                state.parameters["strategy"]["encoder"]["Dense_0"]["kernel"][6:9], 0.0
            )
            adam = state.strategy_optimizer_state[1][0]
            for moment in (adam.mu, adam.nu):
                np.testing.assert_array_equal(
                    moment["encoder"]["Dense_0"]["kernel"][6:9], 0.0
                )
        self.assertEqual(int(state.strategy_updates), 5)
        self.assertEqual(int(state.locomotion_updates), 0)
        equal_trees(
            self, state.parameters["locomotion"], initial.parameters["locomotion"]
        )
        equal_trees(
            self, state.locomotion_optimizer_state, initial.locomotion_optimizer_state
        )
        self.assertFalse(
            np.array_equal(
                state.parameters["strategy"]["pursuer_head"]["command"]["kernel"],
                initial.parameters["strategy"]["pursuer_head"]["command"]["kernel"],
            )
        )
        self.assertFalse(
            np.array_equal(
                state.parameters["strategy"]["evader_head"]["command"]["kernel"],
                initial.parameters["strategy"]["evader_head"]["command"]["kernel"],
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "000000000005"
            save_hierarchical_checkpoint(
                path,
                state,
                update_config=self.config,
                metadata={"constraint": "strategy_previous_command_rows6_9"},
            )
            ordinary = HierarchicalPpoUpdater(
                self.config, policy_config=self.policy_config
            )
            restored, manifest = load_hierarchical_checkpoint(
                path, ordinary.initialize(self.original)
            )
            equal_trees(self, restored, state)
            self.assertEqual(
                manifest["metadata"]["constraint"], "strategy_previous_command_rows6_9"
            )
            actor = jax.jit(
                lambda observation: self.strategy.apply(
                    {"params": restored.parameters["strategy"]}, observation
                )
            )
            equal_trees(
                self,
                actor(self.observation),
                actor(self.observation.at[:, 6:9].set(1.0)),
            )

    def test_launcher_control_exact_delegation_fresh_validation_and_finally_restore(
        self,
    ):
        for masked in (False, True):
            with patch.object(
                launcher.base, "migrate_p2_actor_parameters", return_value=self.original
            ) as initializer:
                updater = launcher.base.HierarchicalPpoUpdater

                def trainer(
                    _config, masked=masked, initializer=initializer, updater=updater
                ):
                    if masked:
                        self.assertIs(
                            launcher.base.HierarchicalPpoUpdater,
                            PreviousCommandConstrainedUpdater,
                        )
                        equal_trees(
                            self,
                            launcher.base.migrate_p2_actor_parameters(None, None),
                            self.parameters,
                        )
                    else:
                        self.assertIs(launcher.base.HierarchicalPpoUpdater, updater)
                        self.assertIs(
                            launcher.base.migrate_p2_actor_parameters, initializer
                        )
                    return "fixture"

                with patch.object(launcher.base, "main", side_effect=trainer):
                    self.assertEqual(
                        launcher.main(
                            launcher.Config(
                                mask_previous_command=masked, freeze_locomotion=True
                            )
                        ),
                        "fixture",
                    )
                self.assertIs(launcher.base.HierarchicalPpoUpdater, updater)
                self.assertIs(launcher.base.migrate_p2_actor_parameters, initializer)
                with (
                    patch.object(
                        launcher.base, "main", side_effect=RuntimeError("fixture")
                    ),
                    self.assertRaises(RuntimeError),
                ):
                    launcher.main(
                        launcher.Config(
                            mask_previous_command=masked, freeze_locomotion=True
                        )
                    )
                self.assertIs(launcher.base.HierarchicalPpoUpdater, updater)
                self.assertIs(launcher.base.migrate_p2_actor_parameters, initializer)
        with self.assertRaises(ValueError):
            launcher.validate_constraint(launcher.Config(freeze_locomotion=False))
        with self.assertRaises(ValueError):
            launcher.validate_constraint(
                launcher.Config(freeze_locomotion=True, restore_checkpoint=Path("old"))
            )


if __name__ == "__main__":
    unittest.main()
