"""CPU-only raw benchmark comparison regressions."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))
SPEC = importlib.util.spec_from_file_location(
    "compare_tag_mechanisms", EVALUATION / "compare_tag_mechanisms.py"
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def fixture(iteration=100, distance=1.5, mode="deterministic"):
    rollouts = [
        {
            "tag_occurred": True,
            "physical_survival": True,
            "terminal_cause": "TAG",
            "role_success": True,
            "initial_distance": distance,
            "final_distance": 0.3,
            "minimum_distance": 0.3,
            "time_below_0_5m": 1.0,
        },
        {
            "tag_occurred": False,
            "physical_survival": True,
            "terminal_cause": "TIME_LIMIT_TRUNCATION",
            "role_success": None,
            "initial_distance": distance,
            "final_distance": 2.0,
            "minimum_distance": 1.0,
            "time_below_0_5m": 0.0,
        },
    ]
    return {
        "passed": True,
        "checkpoint_iteration": iteration,
        "action_mode": mode,
        "config": {
            "seeds": 2,
            "seed": 3000,
            "duration": 10.0,
            "initial_separation": distance,
            "opponent_checkpoint": "historic005",
            "use_self_play": False,
        },
        "scenarios": [
            {"role": role, "rollouts": rollouts} for role in ("pursuer", "evader")
        ],
    }


class ComparisonTests(unittest.TestCase):
    def test_long_study_pairs_intermediate_and_final_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            files = []
            for iteration in (0, 100, 500):
                path = Path(temp) / f"iter_{iteration:012d}" / "probe.json"
                path.parent.mkdir()
                path.write_text(json.dumps(fixture(iteration=iteration)))
                files.append(compare.summarize_file(path, "standard"))
            intermediate = compare.paired_comparisons(files)
            final = compare.paired_comparisons(files, 500)
            self.assertEqual({row["final_iteration"] for row in intermediate}, {100})
            self.assertEqual({row["final_iteration"] for row in final}, {500})
            self.assertTrue(all(row["paired_complete"] for row in intermediate + final))
            missing = compare.paired_comparisons(files[:2], 500)
            self.assertTrue(all(not row["paired_complete"] for row in missing))

    def test_mixed_pairing_stays_separate_and_requires_truthful_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = (
                root
                / "logs/hierarchical/training/tag_motor_noise_control_seed0_job4385"
            )
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"config": {"seed": 0}}))
            for name in compare.expected_mixed_evaluations("4385"):
                path = run / "mixed_evaluation" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = fixture(
                    iteration=int(path.parent.name.removeprefix("iter_")), mode="mixed"
                )
                data.update(
                    {
                        "strategy_action_mode": "stochastic",
                        "motor_action_mode": "deterministic",
                    }
                )
                data["config"].update(
                    {
                        "stochastic_actions": True,
                        "stochastic_locomotion_actions": False,
                        "use_self_play": "selfplay" in name,
                        "initial_separation": 2.0 if "sep2.0" in name else 1.5,
                        "opponent_checkpoint": None
                        if "selfplay" in name
                        else "historic020"
                        if "000000000020" in name
                        else "historic005",
                    }
                )
                path.write_text(json.dumps(data))
            result = compare.compare(root)["runs"][0]
            self.assertTrue(result["mixed_complete"])
            self.assertFalse(result["standard_complete"])
            self.assertEqual(len(result["mixed_comparisons"]), 10)
            self.assertTrue(
                all(row["paired_complete"] for row in result["mixed_comparisons"])
            )
            self.assertTrue(
                all(
                    row["strategy_action_mode"] == "stochastic"
                    and row["motor_action_mode"] == "deterministic"
                    for row in result["mixed_comparisons"]
                )
            )
            one = result["mixed_files"][0]
            reverse = dict(
                one,
                strategy_action_mode="deterministic",
                motor_action_mode="stochastic",
            )
            homogeneous = dict(
                one,
                action_mode="stochastic",
                strategy_action_mode="stochastic",
                motor_action_mode="stochastic",
            )
            self.assertEqual(
                len(compare.paired_comparisons([one, reverse, homogeneous])), 6
            )
            path = Path(one["file"])
            data = json.loads(path.read_text())
            data["action_mode"] = "stochastic"
            path.write_text(json.dumps(data))
            self.assertFalse(compare.summarize_file(path, "mixed")["complete_valid"])
            self.assertFalse(compare.compare(root)["runs"][0]["mixed_complete"])

    def test_candidate_validation_partial_and_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = compare.candidate_validation(root)
            self.assertEqual(len(report["runs"]), 12)
            self.assertFalse(report["all_complete"])
            self.assertEqual(
                sum(run["validation_job"] == "4384" for run in report["runs"]), 4
            )
            name = "tag_pursuit_mechanism_potential050_seed1_job4324"
            source = root / "logs/hierarchical/training" / name
            source.mkdir(parents=True)
            (source / "manifest.json").write_text(json.dumps({"config": {"seed": 1}}))
            directory = root / "logs/hierarchical/candidate_validation/job4375" / name
            directory.mkdir(parents=True)
            for filename in compare.expected_validation("4355"):
                (directory / filename).write_text(json.dumps(fixture()))
            report = compare.candidate_validation(root)
            result = next(run for run in report["runs"] if run["run"] == name)
            self.assertTrue(result["complete"])
            path = next(directory.glob("*.json"))
            data = fixture()
            data["passed"] = False
            path.write_text(json.dumps(data))
            result = next(
                run
                for run in compare.candidate_validation(root)["runs"]
                if run["run"] == name
            )
            self.assertFalse(result["complete"])

    def test_tag_free_rule_excludes_opponent_failures_and_nullable_success(self):
        rows = [
            {
                "tag_occurred": False,
                "physical_survival": True,
                "terminal_cause": "TIME_LIMIT_TRUNCATION",
                "role_success": None,
            },
            {
                "tag_occurred": False,
                "physical_survival": True,
                "terminal_cause": "PURSUER_FALL",
                "role_success": None,
            },
            {
                "tag_occurred": True,
                "physical_survival": True,
                "terminal_cause": "TAG",
                "role_success": True,
            },
            {
                "tag_occurred": False,
                "physical_survival": False,
                "terminal_cause": "TIME_LIMIT_TRUNCATION",
                "role_success": None,
            },
        ]
        result = compare.counts(rows)
        self.assertEqual(result["tag_rate"], 0.25)
        self.assertEqual(result["tag_free_full_window_rate"], 0.25)
        self.assertEqual(result["nullable_role_success"], 3)

    def test_distances_modes_and_selfplay_remain_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            files = []
            for distance in (1.5, 2.0):
                for mode in ("deterministic", "stochastic"):
                    for iteration in (0, 100):
                        path = Path(temp) / f"{distance}_{mode}_{iteration}.json"
                        path.write_text(json.dumps(fixture(iteration, distance, mode)))
                        files.append(compare.summarize_file(path, "standard"))
            pairs = compare.paired_comparisons(files)
            self.assertEqual(len(pairs), 8)
            self.assertTrue(all(row["paired_complete"] for row in pairs))
            self.assertEqual(
                {row["action_mode"] for row in pairs}, {"deterministic", "stochastic"}
            )
            self.assertEqual({row["initial_separation"] for row in pairs}, {1.5, 2.0})
            self.assertEqual(
                pairs[0]["final"]["mean_diagnostics"]["time_below_0_5m"], 0.5
            )
            altered = dict(files[0], panel="self_play")
            self.assertEqual(len(compare.paired_comparisons([files[0], altered])), 4)

    def test_partial_failed_nonfinite_and_missing_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = (
                root
                / "logs/hierarchical/training/tag_heading_diversity_control_seed1_job4355"
            )
            directory = run / "evaluation/iter_000000000100"
            directory.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"config": {"seed": 1}}))
            path = directory / "historical000000000005_sep1.5_stochastic_false.json"
            data = fixture()
            path.write_text(json.dumps(data))
            report = compare.compare(root)
            result = report["runs"][0]
            self.assertEqual(result["training_seed"], 1)
            self.assertFalse(result["standard_complete"])
            self.assertFalse(result["validation_complete"])
            self.assertEqual(len(result["standard_missing"]), 19)
            self.assertFalse(result["comparisons"][0]["paired_complete"])
            self.assertEqual(result["comparisons"][0]["change_final_minus_initial"], {})
            data["passed"] = False
            path.write_text(json.dumps(data))
            self.assertFalse(compare.summarize_file(path, "standard")["complete_valid"])
            data["passed"] = True
            data["scenarios"][0]["rollouts"].pop()
            path.write_text(json.dumps(data))
            self.assertFalse(compare.summarize_file(path, "standard")["complete_valid"])
            data = fixture()
            data["scenarios"][0]["rollouts"][0]["final_distance"] = float("nan")
            path.write_text(json.dumps(data))
            self.assertFalse(compare.summarize_file(path, "standard")["finite"])

    def test_exact_standard_path_set_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = (
                root
                / "logs/hierarchical/training/tag_pursuit_mechanism_control_seed1_job4324"
            )
            for name in compare.expected_evaluations():
                path = run / "evaluation" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                iteration = int(path.parent.name.removeprefix("iter_"))
                path.write_text(json.dumps(fixture(iteration=iteration)))
            (run / "manifest.json").write_text(json.dumps({"config": {"seed": 1}}))
            result = compare.compare(root)["runs"][0]
            self.assertTrue(result["standard_complete"])
            self.assertIsNone(result["validation_complete"])
            extra = run / "evaluation/iter_000000000100/unexpected.json"
            extra.write_text(json.dumps(fixture()))
            self.assertFalse(compare.compare(root)["runs"][0]["standard_complete"])


if __name__ == "__main__":
    unittest.main()
