"""CPU-only anchored benchmark-rating tests; no simulator or plotting imports."""

import copy
import math
import unittest
from types import SimpleNamespace

from motionforge.training.tag_benchmark_ratings import PROTOCOL, TagBenchmarkRatings


def calibration():
    return {
        "known": {
            "pursuer": {"failures": 20, "episodes": 100},
            "evader": {"failures": 60, "episodes": 100},
        },
        "hard": {
            "pursuer": {"failures": 90, "episodes": 100},
            "evader": {"failures": 80, "episodes": 100},
        },
    }


class AnchoredRatingTests(unittest.TestCase):
    def test_training_episode_counts_include_all_outcomes_not_padding(self):
        import jax.numpy as jp

        from motionforge.training.tag_metrics import hierarchical_tag_training_metrics

        batch = SimpleNamespace(
            transition_valid=jp.asarray(
                [True, True, True, True, True, True, False, False]
            ),
            transitions=SimpleNamespace(
                environment_id=jp.repeat(jp.arange(4), 2),
                role=jp.repeat(
                    jp.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
                    2,
                    axis=0,
                ),
                tagged=jp.zeros(8, bool),
                timed_out=jp.zeros(8, bool),
                fallen=jp.zeros((8, 2), bool),
                out_of_bounds=jp.zeros((8, 2), bool),
            ),
            strategy=SimpleNamespace(
                reward=jp.zeros(8), value=jp.zeros(8), valid=jp.ones(8, bool)
            ),
        )
        metrics = hierarchical_tag_training_metrics(batch)
        self.assertEqual(int(metrics["strategy/pursuer/episodes"]), 2)
        self.assertEqual(int(metrics["strategy/evader/episodes"]), 1)
        self.assertEqual(float(metrics["strategy/pursuer/tag_rate"]), 0)

    def test_fixed_reference_formula_and_no_extra_role_bias(self):
        tracker = TagBenchmarkRatings(calibration())
        initial = tracker.metrics("known")
        self.assertEqual(PROTOCOL, "anchored_online_finite_window_v1")
        self.assertEqual(initial["benchmark/pursuer/relative_skill"], 0.5)
        self.assertEqual(
            initial["benchmark/pursuer/relative_skill_has_observations"], 0
        )
        self.assertAlmostEqual(
            initial["benchmark/pursuer/predicted_completion"], 80.5 / 101
        )
        self.assertAlmostEqual(
            initial["benchmark/evader/predicted_completion"], 40.5 / 101
        )
        result = tracker.update("known", "pursuer", 1.0, 16)
        expected_delta = 24 / tracker.rating_scale * (1 - 80.5 / 101)
        self.assertAlmostEqual(tracker.learner_abilities["pursuer"], expected_delta)
        self.assertAlmostEqual(
            result["benchmark/pursuer/predicted_completion"], 80.5 / 101
        )
        self.assertGreater(
            result["benchmark/pursuer/predicted_completion_after_update"], 80.5 / 101
        )

    def test_fixed_difficulty_not_inverse_skill_or_mutable_caller_table(self):
        table = calibration()
        tracker = TagBenchmarkRatings(table)
        before = tracker.metrics("known")
        for _ in range(20):
            tracker.update("known", "pursuer", 1.0, 16)
        table["known"]["pursuer"]["failures"] = 99
        after = tracker.metrics("known")
        self.assertEqual(after["benchmark/pursuer/opponent_difficulty"], 0.2)
        self.assertEqual(
            after["benchmark/pursuer/opponent_challenge_logit"],
            before["benchmark/pursuer/opponent_challenge_logit"],
        )
        self.assertNotAlmostEqual(
            after["benchmark/pursuer/opponent_difficulty"],
            1 - after["benchmark/pursuer/relative_skill"],
        )
        self.assertGreater(
            after["benchmark/pursuer/relative_skill"],
            before["benchmark/pursuer/relative_skill"],
        )

    def test_zero_tags_still_updates_and_roles_do_not_cancel(self):
        tracker = TagBenchmarkRatings(calibration())
        result = tracker.update_from_training_metrics(
            "known",
            {
                "strategy/pursuer/tag_rate": 0.0,
                "termination/evader/time_limit_truncation_rate": 1.0,
            },
            {"pursuer": 2048, "evader": 2048},
        )
        self.assertEqual(result["benchmark/pursuer/updated"], 1)
        self.assertLess(result["benchmark/pursuer/relative_skill"], 0.5)
        self.assertGreater(result["benchmark/evader/relative_skill"], 0.5)
        self.assertEqual(result["benchmark/pursuer/rated_episodes"], 2048)
        self.assertFalse(any("combined" in key for key in result))
        improving = TagBenchmarkRatings(calibration())
        for role in ("pursuer", "evader"):
            improving.update("known", role, 1.0, 16)
        snapshot = improving.metrics("known")
        self.assertGreater(snapshot["benchmark/pursuer/relative_skill"], 0.5)
        self.assertGreater(snapshot["benchmark/evader/relative_skill"], 0.5)

    def test_role_independence_and_single_role_exposure(self):
        tracker = TagBenchmarkRatings(calibration())
        before = tracker.metrics("known")
        result = tracker.update_from_training_metrics(
            "known", {"strategy/pursuer/tag_rate": 0.0}, {"pursuer": 16}
        )
        self.assertEqual(result["benchmark/evader/skipped_zero_episodes"], 1)
        after = tracker.metrics("known")
        for key in before:
            if key.startswith("benchmark/evader/"):
                self.assertEqual(after[key], before[key])

    def test_unknown_calibration_is_gap_with_explicit_coverage(self):
        tracker = TagBenchmarkRatings(calibration())
        result = tracker.update("unknown", "pursuer", 0.75, 16)
        self.assertNotIn("benchmark/pursuer/opponent_difficulty", result)
        self.assertNotIn("benchmark/pursuer/predicted_completion", result)
        self.assertNotIn("benchmark/pursuer/relative_skill", result)
        self.assertEqual(result["benchmark/pursuer/updated"], 0)
        self.assertEqual(result["benchmark/pursuer/skipped_uncalibrated"], 1)
        self.assertEqual(result["benchmark/pursuer/uncalibrated_episodes"], 16)
        self.assertEqual(result["benchmark/pursuer/calibration_coverage"], 0)
        rated = tracker.update("known", "pursuer", 1.0, 16)
        previous = tracker.learner_abilities.copy()
        result = tracker.update("unknown", "pursuer", 1.0, 16)
        self.assertEqual(tracker.learner_abilities, previous)
        self.assertEqual(
            result["benchmark/pursuer/relative_skill"],
            rated["benchmark/pursuer/relative_skill"],
        )
        self.assertAlmostEqual(result["benchmark/pursuer/calibration_coverage"], 1 / 3)
        self.assertNotIn("benchmark/pursuer/opponent_difficulty", result)

    def test_no_payoff_changes_or_opponent_failure_completion(self):
        tracker = TagBenchmarkRatings(calibration())
        metrics = {
            "strategy/pursuer/tag_rate": 0.0,
            "termination/evader/time_limit_truncation_rate": 0.0,
            "strategy/evader/opponent_out_of_bounds_rate": 1.0,
            "reward/pursuer/tag": 1.0,
            "reward/evader/tag": -1.0,
        }
        original = copy.deepcopy(metrics)
        result = tracker.update_from_training_metrics(
            "known", metrics, {"pursuer": 16, "evader": 16}
        )
        self.assertEqual(metrics, original)
        self.assertEqual(result["benchmark/pursuer/observed_completion"], 0)
        self.assertEqual(result["benchmark/evader/observed_completion"], 0)
        self.assertLess(result["benchmark/evader/ability_delta"], 0)

    def test_zero_episode_skip_missing_metric_validation_and_atomic_input(self):
        tracker = TagBenchmarkRatings(calibration())
        before = tracker.metrics("known")
        result = tracker.update("known", "pursuer", None, 0)
        self.assertEqual(result["benchmark/pursuer/skipped_zero_episodes"], 1)
        self.assertEqual(tracker.metrics("known"), before)
        with self.assertRaises(ValueError):
            tracker.update_from_training_metrics(
                "known",
                {"strategy/pursuer/tag_rate": 1.0},
                {"pursuer": 16, "evader": 16},
            )
        self.assertEqual(tracker.metrics("known"), before)
        for rate in (float("nan"), float("inf"), -0.1, 1.1, None):
            with self.assertRaises(ValueError):
                tracker.update("known", "pursuer", rate, 1)
        for episodes in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                tracker.update("known", "pursuer", 0.5, episodes)
        with self.assertRaises(ValueError):
            tracker.update("known", "invalid", 0.5, 1)

    def test_extreme_calibration_bounded_and_zero_exposure_gap(self):
        tracker = TagBenchmarkRatings(
            {
                "extreme": {
                    "pursuer": {"failures": 0, "episodes": 16},
                    "evader": {"failures": 16, "episodes": 16},
                },
                "empty": {"pursuer": {"failures": 0, "episodes": 0}},
            }
        )
        self.assertNotIn(
            "benchmark/pursuer/opponent_difficulty", tracker.metrics("empty")
        )
        for i in range(500):
            for role in ("pursuer", "evader"):
                result = tracker.update("extreme", role, float(i % 2), 16)
                self.assertTrue(
                    all(
                        isinstance(value, (int, float)) and math.isfinite(value)
                        for value in result.values()
                    )
                )
                for key in (
                    "relative_skill",
                    "opponent_difficulty",
                    "predicted_completion",
                    "predicted_completion_after_update",
                ):
                    self.assertTrue(0 <= result[f"benchmark/{role}/{key}"] <= 1)
        self.assertEqual(
            tracker.metrics("extreme")["benchmark/pursuer/opponent_difficulty"], 0
        )
        self.assertEqual(
            tracker.metrics("extreme")["benchmark/evader/opponent_difficulty"], 1
        )

    def test_bad_calibration_rejected(self):
        for counts in (
            {"failures": 2, "episodes": 1},
            {"failures": -1, "episodes": 1},
            {"failures": 0, "episodes": 1.5},
            {"failures": 0},
            {"failures": True, "episodes": 1},
        ):
            with self.assertRaises(ValueError):
                TagBenchmarkRatings({"known": {"pursuer": counts}})
        for kwargs in ({"learning_rate": float("nan")}, {"rating_scale": 0.0}):
            with self.assertRaises(ValueError):
                TagBenchmarkRatings(calibration(), **kwargs)


if __name__ == "__main__":
    unittest.main()
