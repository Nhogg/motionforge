"""CPU-only monitor tests; no simulation dependencies."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "monitor_tag_mechanism",
    Path(__file__).resolve().parents[1] / "evaluation/monitor_tag_mechanism.py",
)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def test_long_study_requires_all_three_phases_and_reserved_validation(self):
        paths = monitor.expected_evaluations("4415")
        self.assertEqual(len(paths), 30)
        for iteration in (0, 100, 500):
            self.assertEqual(
                sum(path.startswith(f"iter_{iteration:012d}/") for path in paths), 10
            )
        self.assertEqual(len(monitor.expected_validation("4415")), 8)
        self.assertEqual(len(monitor.expected_evaluations()), 20)

    def test_motor_array_expected_paths_and_truthful_mixed_modes(self):
        paths = monitor.expected_mixed_evaluations("4385")
        self.assertEqual(len(paths), 10)
        self.assertEqual(
            sum(path.startswith("iter_000000000000/") for path in paths), 5
        )
        self.assertEqual(
            sum(path.startswith("iter_000000000100/") for path in paths), 5
        )
        self.assertEqual(monitor.expected_mixed_evaluations("4355"), set())
        data = {
            "action_mode": "mixed",
            "strategy_action_mode": "stochastic",
            "motor_action_mode": "deterministic",
            "config": {
                "stochastic_actions": True,
                "stochastic_locomotion_actions": False,
            },
        }
        metadata = monitor.sampling_metadata(data)
        self.assertTrue(metadata["action_mode_metadata_consistent"])
        self.assertEqual(metadata["action_mode"], "mixed")
        data["action_mode"] = "stochastic"
        self.assertFalse(
            monitor.sampling_metadata(data)["action_mode_metadata_consistent"]
        )
        self.assertEqual(
            monitor.sampling_metadata(data)["motor_action_mode"], "deterministic"
        )
        data["action_mode"] = "mixed"
        data["config"]["stochastic_locomotion_actions"] = None
        self.assertFalse(
            monitor.sampling_metadata(data)["action_mode_metadata_consistent"]
        )
        self.assertEqual(
            monitor.sampling_metadata(data)["motor_action_mode"], "stochastic"
        )

    def test_streamed_window_partial_line_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            rows = [
                {
                    "metrics": {
                        "training/iteration": index,
                        "training/locomotion_updates": 0,
                        "pursuer/distance_closed": index,
                    },
                    "metrics_finite": True,
                }
                for index in range(25)
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + '\n{"metrics":'
            )
            result = monitor.training_history(path)
            self.assertEqual(result["recent_records"], 20)
            self.assertEqual(
                result["recent_mean_variable_opponents"]["pursuer/distance_closed"],
                14.5,
            )
            self.assertEqual(result["invalid_or_partial_lines"], 1)
            self.assertTrue(result["all_metrics_finite"])
            self.assertEqual(result["max_locomotion_updates"], 0)
            rows[0]["metrics"]["bad"] = float("nan")
            path.write_text("\n".join(json.dumps(row) for row in rows))
            self.assertFalse(monitor.training_history(path)["all_metrics_finite"])

    def test_evaluation_counts_tags_not_nullable_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evaluation.json"
            path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "scenarios": [
                            {
                                "role": "pursuer",
                                "role_success_rate": 1,
                                "rollouts": [
                                    {
                                        "role_success": None,
                                        "tag_occurred": False,
                                        "physical_survival": True,
                                        "terminal_cause": "TIME_LIMIT_TRUNCATION",
                                        "terminated": False,
                                        "truncated": True,
                                    },
                                    {
                                        "role_success": True,
                                        "tag_occurred": True,
                                        "physical_survival": True,
                                        "terminal_cause": "TAG",
                                        "terminated": True,
                                        "truncated": False,
                                    },
                                ],
                            }
                        ],
                    }
                )
            )
            role = monitor.evaluation(path)["roles"][0]
            self.assertEqual(role["tag_rate"], 0.5)
            self.assertEqual(role["physical_survival_rate"], 1)
            self.assertEqual(role["nullable_role_success_observations"], 1)

    def test_missing_data_is_not_training_or_evaluation_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = (
                root
                / "logs/hierarchical/training/tag_initialization_init001_seed1_job4340"
            )
            run.mkdir(parents=True)
            result = monitor.snapshot(root)["runs"][0]
            self.assertFalse(result["training_complete"])
            self.assertFalse(result["evaluations_complete"])
            self.assertIsNone(result["freeze_verified_zero_updates"])
            self.assertIsNone(result["training"]["all_metrics_finite"])
            self.assertEqual(len(result["evaluation_missing"]), 20)

    def test_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested/status.md"
            monitor.atomic_write(path, "first")
            monitor.atomic_write(path, "second")
            self.assertEqual(path.read_text(), "second")
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_validation_missing_failed_and_complete_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = (
                root
                / "logs/hierarchical/training/tag_heading_diversity_control_seed1_job4355"
            )
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "initial_heading_noise_radians": 1.57,
                            "historical_opponent_checkpoints": ["training_reference"],
                        }
                    }
                )
            )
            result = monitor.snapshot(root)["runs"][0]
            self.assertTrue(result["validation_required"])
            self.assertFalse(result["validation_complete"])
            self.assertEqual(len(result["validation_missing"]), 8)
            self.assertEqual(result["config"]["initial_heading_noise_radians"], 1.57)
            validation = run / "validation"
            validation.mkdir()
            expected = sorted(monitor.expected_validation("4355"))
            for name in expected:
                (validation / name).write_text(json.dumps({"passed": True}))
            (validation / expected[0]).write_text(json.dumps({"passed": False}))
            result = monitor.snapshot(root)["runs"][0]
            self.assertEqual(result["validation_missing"], [])
            self.assertFalse(result["validation_complete"])
            (validation / expected[0]).write_text(json.dumps({"passed": True}))
            result = monitor.snapshot(root)["runs"][0]
            self.assertTrue(result["validation_complete"])
            self.assertFalse(result["evaluations_complete"])
            self.assertEqual(result["evaluation_files"], 0)

    def test_config_provenance_and_action_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evaluation.json"
            for stochastic in (False, True):
                path.write_text(
                    json.dumps(
                        {
                            "passed": True,
                            "action_mode": "stochastic"
                            if stochastic
                            else "deterministic",
                            "config": {
                                "initial_heading_noise_radians": 1.57,
                                "stochastic_actions": stochastic,
                                "use_self_play": False,
                                "opponent_checkpoint": "reserved_reference",
                            },
                        }
                    )
                )
                result = monitor.evaluation(
                    path, {"historical_opponent_checkpoints": ["training_reference"]}
                )
                self.assertEqual(result["initial_heading_noise_radians"], 1.57)
                self.assertEqual(result["stochastic_actions"], stochastic)
                self.assertEqual(
                    result["action_mode"],
                    "stochastic" if stochastic else "deterministic",
                )
                self.assertTrue(result["opponent_not_in_training_historical_panel"])
                self.assertFalse(result["use_self_play"])
                result = monitor.evaluation(
                    path, {"historical_opponent_checkpoints": ["reserved_reference"]}
                )
                self.assertFalse(result["opponent_not_in_training_historical_panel"])
            self.assertEqual(monitor.expected_validation("4340"), set())
            self.assertIsNone(
                monitor.evaluation(path)["opponent_not_in_training_historical_panel"]
            )


if __name__ == "__main__":
    unittest.main()
