"""Regression tests for cancelling aggregates and misleading reward labels."""

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from motionforge.callbacks.tag_progress import TagProgressPlotCallback, progress_series


class ProgressTests(unittest.TestCase):
    def test_legacy_flat_elo_is_not_substituted_for_measurement(self):
        history = [
            {
                "environment_steps": 10,
                "strategy/pursuer/episode_return_mean": -0.3,
                "elo/pursuer/perceived_skill": 0.5,
                "elo/pursuer/task_difficulty": 0.5,
            }
        ]
        series = progress_series(history, "pursuer")
        self.assertEqual(series["reward"], [-0.3])
        self.assertTrue(math.isnan(series["skill"][0]))
        self.assertTrue(math.isnan(series["difficulty"][0]))
        with self.assertRaisesRegex(ValueError, "role separated"):
            progress_series(history, "combined")

    def test_corrected_series_and_role_separated_overview_are_rendered(self):
        captured = []
        logger = SimpleNamespace(log_image=lambda **values: captured.append(values))
        metrics = {}
        for role, reward, skill, difficulty in (
            ("pursuer", -0.2, 0.7, 0.3),
            ("evader", -0.8, 0.8, 0.6),
        ):
            metrics.update(
                {
                    f"strategy/{role}/episode_return_mean": reward,
                    f"benchmark/{role}/relative_skill": skill,
                    f"benchmark/{role}/opponent_difficulty": difficulty,
                    f"benchmark/{role}/observed_completion": 0.5,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            callback = TagProgressPlotCallback(logger, Path(directory))
            callback(100, metrics)
            self.assertEqual(len(captured), 3)
            self.assertTrue(
                all(item["path"].stat().st_size > 1000 for item in captured)
            )
            self.assertEqual(
                progress_series(callback.history, "pursuer")["reward"], [-0.2]
            )
            self.assertEqual(
                progress_series(callback.history, "evader")["skill"], [0.8]
            )
            self.assertTrue(all("not 1 minus" in item["caption"] for item in captured))

    def test_uncalibrated_trainer_reports_missing_not_measured_half(self):
        path = (
            Path(__file__).resolve().parents[1] / "training/train_hierarchical_tag.py"
        )
        spec = importlib.util.spec_from_file_location(
            "tag_progress_trainer_fixture", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        tracker, group = module.benchmark_tracker(module.Config(), SimpleNamespace())
        self.assertIsNone(group)
        metrics = tracker.update("unknown", "pursuer", 0.0, 32)
        self.assertNotIn("benchmark/pursuer/opponent_difficulty", metrics)
        self.assertNotIn("benchmark/pursuer/relative_skill", metrics)
        self.assertEqual(metrics["benchmark/pursuer/observed_completion"], 0)


if __name__ == "__main__":
    unittest.main()
