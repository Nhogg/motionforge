"""Check actual strategy PPO gradients against a full valid-sample objective."""

import unittest

import jax
import jax.numpy as jp
import numpy as np

from motionforge.models import tanh_normal_log_prob
from motionforge.models.hierarchical import StrategyModule
from motionforge.rollout import AdvantageTargets, StrategyBatch
from motionforge.training.hierarchical_ppo import (
    PpoLossConfig,
    locomotion_ppo_loss,
    strategy_ppo_loss,
)
from motionforge.training.hierarchical_update import (
    HierarchicalPpoUpdater,
    HierarchicalUpdateConfig,
)
from motionforge.training.tag_strategy_objective_ablation import (
    FullBatchStrategyObjectiveUpdater,
)


class TestAccumulation(unittest.TestCase):
    def setUp(self):
        n = 8
        x = jax.random.normal(jax.random.PRNGKey(12), (n, 11)) * 0.1
        x = x.at[:, -2:].set(jp.tile(jp.array([[1.0, 0.0], [0.0, 1.0]]), (4, 1)))
        module = StrategyModule((128, 128), (1.0, 0.5, 1.0), -1.0)
        self.parameters = module.init(jax.random.PRNGKey(5), x)["params"]
        output = module.apply({"params": self.parameters}, x)
        command = jp.clip(
            output.command + jax.random.normal(jax.random.PRNGKey(7), (n, 3)) * 0.04,
            -jp.array([0.99, 0.49, 0.99]),
            jp.array([0.99, 0.49, 0.99]),
        )
        zeros = jp.zeros(n)
        valid = jp.array([True, False, False, False, True, True, True, True])
        self.batch = StrategyBatch(
            observation=x,
            command=command,
            value=output.value,
            log_probability=tanh_normal_log_prob(
                command, output.location, output.log_std, jp.array([1.0, 0.5, 1.0])
            ),
            entropy=zeros,
            reward=zeros,
            done=jp.zeros(n, bool),
            valid=valid,
            role=x[:, -2:],
            episode_id=jp.arange(n),
            environment_id=jp.arange(n),
        )
        self.targets = AdvantageTargets(
            advantages=jp.array([2.0, 0, 0, 0, -1.0, -0.5, 0.5, 1.0]),
            returns=output.value + jp.array([2.0, 0, 0, 0, -1.0, -0.5, 0.5, 1.0]),
        )

    def gradients(self, updater, config):
        return updater._loss_and_grad(
            strategy_ppo_loss, self.parameters, self.batch, self.targets, config
        )

    def compare(self, valid, normalize):
        self.batch = self.batch.replace(valid=valid)
        config = PpoLossConfig(normalize_advantages=normalize)
        full = HierarchicalPpoUpdater(HierarchicalUpdateConfig(minibatch_size=None))
        expected_metrics, expected = self.gradients(full, config)
        for size in (2, 4, 8):
            updater = FullBatchStrategyObjectiveUpdater(
                HierarchicalUpdateConfig(minibatch_size=size)
            )
            actual_metrics, actual = self.gradients(updater, config)
            for a, b in zip(
                jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
            ):
                np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
            for a, b in zip(
                jax.tree.leaves(actual_metrics),
                jax.tree.leaves(expected_metrics),
                strict=True,
            ):
                np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
            loss, _metrics = updater._loss(
                strategy_ppo_loss, self.parameters, self.batch, self.targets, config
            )
            np.testing.assert_allclose(loss, expected_metrics.total_loss, atol=2e-6)

    def test_uneven_padding_global_normalization(self):
        self.compare(self.batch.valid, True)

    def test_uneven_padding_no_normalization(self):
        self.compare(self.batch.valid, False)

    def test_empty_chunks(self):
        self.compare(jp.array([False] * 4 + [True] * 4), True)

    def test_all_invalid(self):
        self.compare(jp.zeros(8, bool), True)

    def test_perturbed_policy_and_clipping(self):
        self.parameters = jax.tree.map(lambda x: x * 1.4 + 0.03, self.parameters)
        self.compare(self.batch.valid, True)

    def test_locomotion_untouched(self):
        updater = FullBatchStrategyObjectiveUpdater(
            HierarchicalUpdateConfig(minibatch_size=4)
        )
        config = PpoLossConfig()
        function, batch, targets, returned = updater._prepare_strategy(
            locomotion_ppo_loss, self.batch, self.targets, config
        )
        self.assertIs(function, locomotion_ppo_loss)
        self.assertIs(batch, self.batch)
        self.assertIs(targets, self.targets)
        self.assertIs(returned, config)

    def test_jitted_gradient(self):
        config = PpoLossConfig()
        updater = FullBatchStrategyObjectiveUpdater(
            HierarchicalUpdateConfig(minibatch_size=4)
        )
        expected_metrics, expected = self.gradients(updater, config)
        metrics, actual = jax.jit(
            lambda p: updater._loss_and_grad(
                strategy_ppo_loss, p, self.batch, self.targets, config
            )
        )(self.parameters)
        for a, b in zip(
            jax.tree.leaves((metrics, actual)),
            jax.tree.leaves((expected_metrics, expected)),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)

    def test_control_unchanged(self):
        config = PpoLossConfig()
        original = HierarchicalPpoUpdater(HierarchicalUpdateConfig(minibatch_size=4))
        control = FullBatchStrategyObjectiveUpdater(
            HierarchicalUpdateConfig(minibatch_size=4), accumulation_mode="control"
        )
        ma, ga = self.gradients(original, config)
        mb, gb = self.gradients(control, config)
        for a, b in zip(
            jax.tree.leaves((ma, ga)), jax.tree.leaves((mb, gb)), strict=True
        ):
            np.testing.assert_array_equal(a, b)

    def test_original_differs_on_uneven_valid_counts(self):
        config = PpoLossConfig(normalize_advantages=False)
        full = HierarchicalPpoUpdater(HierarchicalUpdateConfig(minibatch_size=None))
        chunked = HierarchicalPpoUpdater(HierarchicalUpdateConfig(minibatch_size=4))
        _, a = self.gradients(full, config)
        _, b = self.gradients(chunked, config)
        difference = max(
            float(jp.max(jp.abs(x - y)))
            for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
        )
        self.assertGreater(difference, 1e-3)
        print("Confirmed original chunk-weighting gradient discrepancy:", difference)


if __name__ == "__main__":
    unittest.main()
