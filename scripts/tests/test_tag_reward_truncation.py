"""CPU regressions for TAG payoff, shaping, ending precedence and GAE."""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np

from motionforge.envs.tag_endings import TagEndCause, classify_tag_ending
from motionforge.rollout import AdvantageConfig, generalized_advantage_estimate
from motionforge.rollout.tag_reward import (
    TagProgressConfig,
    TagRewardConfig,
    tag_strategy_reward_terms,
)
from motionforge.rollout.trajectory import sparse_tag_strategy_reward


class RewardTests(unittest.TestCase):
    def terms(self, role=True, previous=2.0, current=2.0, **overrides):
        args = {
            "previous_distance": jp.asarray(previous),
            "current_distance": jp.asarray(current),
            "command": jp.zeros(3),
            "previous_command": jp.zeros(3),
            "position": jp.zeros(2),
            "arena_half_extent": 4.0,
            "tagged": jp.asarray(False),
            "own_fall": jp.asarray(False),
            "own_oob": jp.asarray(False),
            "learner_is_pursuer": role,
            "config": TagRewardConfig(),
        }
        return tag_strategy_reward_terms(**(args | overrides))

    def test_tag_payoff(self):
        for role, sign in ((True, 1.0), (False, -1.0)):
            term = self.terms(role, tagged=jp.asarray(True))
            self.assertEqual(float(term.tag), sign)
            self.assertEqual(float(term.total), sign)
            terminal = SimpleNamespace(tagged=jp.asarray(True))
            self.assertEqual(
                float(
                    sparse_tag_strategy_reward(
                        termination=terminal, learner_index=0, learner_is_pursuer=role
                    )
                ),
                sign,
            )

    def test_progress_symmetry_sign_and_no_clipping(self):
        for previous, current in [(2, 2), (2, 1.98), (2, 2.02), (10, 0), (0, 10)]:
            p, e = (
                self.terms(True, previous, current),
                self.terms(False, previous, current),
            )
            np.testing.assert_allclose(p.progress, -e.progress, atol=1e-8)
            expected = 0.25 * (previous - current) / (8.0 * np.sqrt(2.0))
            np.testing.assert_allclose(p.progress, expected, atol=1e-7)
            self.assertEqual(np.sign(float(p.progress)), np.sign(previous - current))

    def test_oscillation_telescopes_without_free_reward(self):
        # Seven small closing steps, then one large retreat. Independent
        # clipping would farm positive reward on each such closed cycle.
        cycle = np.asarray([8.,7.,6.,5.,4.,3.,2.,1.])
        distances = np.append(np.tile(cycle,100),8.)
        terms = self.terms(
            previous=jp.asarray(distances[:-1]), current=jp.asarray(distances[1:])
        )
        np.testing.assert_allclose(jp.sum(terms.progress), 0.0, atol=1e-5)
        self.assertGreater(float(jp.max(jp.abs(terms.progress))), 0.02)

    def test_discounted_potential_formula_and_telescoping(self):
        distances = jp.asarray([2.0, 1.0, 4.0, 3.0])
        gamma = 0.9
        terms = self.terms(
            previous=distances[:-1], current=distances[1:], discount=gamma
        )
        phi = -0.25 * distances / (8.0 * np.sqrt(2.0))
        np.testing.assert_allclose(
            terms.progress, gamma * phi[1:] - phi[:-1], atol=1e-7
        )
        weights = gamma ** jp.arange(3)
        np.testing.assert_allclose(
            jp.sum(weights * terms.progress), gamma**3 * phi[-1] - phi[0], atol=1e-7
        )
        evader = self.terms(
            False, previous=distances[:-1], current=distances[1:], discount=gamma
        )
        np.testing.assert_allclose(evader.progress, -terms.progress, atol=1e-8)

    def test_disabled_and_fixed_normalization(self):
        disabled = TagRewardConfig(progress=TagProgressConfig(enabled=False))
        self.assertEqual(
            float(self.terms(previous=4.0, current=1.0, config=disabled).progress), 0.0
        )
        fixed = TagRewardConfig(
            progress=TagProgressConfig(alpha=0.25, distance_normalization_m=10.0)
        )
        np.testing.assert_allclose(
            self.terms(previous=4.0, current=1.0, config=fixed).progress, 0.075
        )

    def test_failure_is_own_penalty_not_tag(self):
        for role in (True, False):
            for key in ("own_fall", "own_oob"):
                terms = self.terms(role, **{key: jp.asarray(True)})
                self.assertEqual(float(terms.tag), 0.0)
                self.assertEqual(float(terms.total), -1.0)

    def test_symmetric_boundary_and_smoothness(self):
        args = {
            "position": jp.asarray([3.5, 0.0]),
            "command": jp.asarray([1.0, 0.5, 0.0]),
        }
        p, e = self.terms(True, **args), self.terms(False, **args)
        self.assertEqual(float(p.boundary), float(e.boundary))
        self.assertEqual(float(p.command_smoothness), float(e.command_smoothness))
        np.testing.assert_allclose(p.command_smoothness, -0.005 * 1.25)


class EndingTests(unittest.TestCase):
    def ending(self, **overrides):
        args = {
            "contact": jp.asarray(False),
            "fallen": jp.zeros(2, bool),
            "out_of_bounds": jp.zeros(2, bool),
            "nonfinite": jp.asarray(False),
            "time_limit": jp.asarray(False),
        }
        result = classify_tag_ending(**(args | overrides))
        self.assertFalse(bool(result[1] & result[2]))
        return result

    def test_timeout_has_no_game_payoff(self):
        tag, terminated, truncated, cause = self.ending(time_limit=jp.asarray(True))
        self.assertFalse(bool(tag))
        self.assertFalse(bool(terminated))
        self.assertTrue(bool(truncated))
        self.assertEqual(int(cause), TagEndCause.TIME_LIMIT_TRUNCATION)
        for role in (True, False):
            self.assertEqual(
                float(
                    sparse_tag_strategy_reward(
                        termination=SimpleNamespace(tagged=tag),
                        learner_index=0,
                        learner_is_pursuer=role,
                    )
                ),
                0.0,
            )

    def test_true_terminal_beats_timeout_and_failures_invalidate_contact(self):
        tag, terminal, truncated, cause = self.ending(
            contact=jp.asarray(True), time_limit=jp.asarray(True)
        )
        self.assertTrue(bool(tag))
        self.assertTrue(bool(terminal))
        self.assertFalse(bool(truncated))
        self.assertEqual(int(cause), TagEndCause.TAG)
        for field, flags, expected in [
            ("fallen", [1, 0], TagEndCause.PURSUER_FALL),
            ("fallen", [0, 1], TagEndCause.EVADER_FALL),
            ("out_of_bounds", [1, 0], TagEndCause.PURSUER_OOB),
            ("out_of_bounds", [0, 1], TagEndCause.EVADER_OOB),
        ]:
            tag, terminal, truncated, cause = self.ending(
                contact=jp.asarray(True),
                time_limit=jp.asarray(True),
                **{field: jp.asarray(flags, bool)},
            )
            self.assertFalse(bool(tag))
            self.assertTrue(bool(terminal))
            self.assertFalse(bool(truncated))
            self.assertEqual(int(cause), expected)
        self.assertEqual(
            int(self.ending(nonfinite=jp.asarray(True), contact=jp.asarray(True))[3]),
            TagEndCause.NONFINITE,
        )


class BootstrapTests(unittest.TestCase):
    def gae(self, rewards, values, terminated, truncated, valid, bootstrap, **kwargs):
        return jax.jit(
            lambda: generalized_advantage_estimate(
                jp.asarray(rewards, dtype=jp.float32),
                jp.asarray(values, dtype=jp.float32),
                jp.asarray(terminated, bool),
                jp.asarray(valid, bool),
                jp.asarray(bootstrap, dtype=jp.float32),
                AdvantageConfig(discount=1.0, gae_lambda=1.0),
                truncated=jp.asarray(truncated, bool),
                **kwargs,
            )
        )()

    def test_truncation_bootstraps_and_termination_does_not(self):
        np.testing.assert_allclose(
            self.gae([0], [2], [False], [True], [True], 7).returns, [7]
        )
        np.testing.assert_allclose(
            self.gae([1], [2], [True], [False], [True], 7).returns, [1]
        )

    def test_padding_preserves_actual_final_state_value(self):
        result = self.gae(
            [0.1, 0.2, 999, 999],
            [1, 2, 100, 100],
            [False] * 4,
            [False, True, False, False],
            [True, True, False, False],
            7,
        )
        np.testing.assert_allclose(result.returns, [7.3, 7.2, 0, 0], atol=1e-6)

    def test_trace_does_not_leak_across_truncated_reset(self):
        result = self.gae(
            [0, 100],
            [1, 2],
            [False, True],
            [True, False],
            [True, True],
            0,
            next_values=jp.asarray([7.0, 0.0]),
        )
        np.testing.assert_allclose(result.returns, [7, 100])


if __name__ == "__main__":
    unittest.main()
