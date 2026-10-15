"""CPU fixture tests for calibration provenance and honest history replay."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from motionforge.evaluation.tag_progress_calibration import (
    COLLECTION_PROTOCOL,
    select_calibration,
)

DIRECTORY = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "rebuild_tag_progress", DIRECTORY / "rebuild_tag_progress.py"
)
rebuild = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rebuild)


def setup_references(root):
    for number in rebuild.REFERENCES:
        checkpoint = root / rebuild.HISTORICAL / f"{number:012d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text(
            json.dumps({"state_sha256": f"{number:064x}"})
        )


def calibration_file(root, number=5, motor=True, heading=0.0):
    checkpoint = root / rebuild.HISTORICAL / f"{number:012d}"
    rows = [
        {
            "seed": seed,
            "tag_occurred": False,
            "physical_survival": True,
            "terminal_cause": "TIME_LIMIT_TRUNCATION",
            "role_success": None,
        }
        for seed in range(7000, 7032)
    ]
    return {
        "passed": True,
        "backend": "gpu",
        "evaluation_protocol": "tag_distance_potential_truncation_v2",
        "checkpoint": str(root / rebuild.ANCHOR),
        "checkpoint_iteration": 950,
        "action_mode": "stochastic" if motor else "mixed",
        "strategy_action_mode": "stochastic",
        "motor_action_mode": "stochastic" if motor else "deterministic",
        "checks": {
            key: True
            for key in (
                "all_rollouts_valid",
                "finite_metrics",
                "gpu_backend",
                "parameters_loaded",
                "role_count",
            )
        },
        "config": {
            "checkpoint": str(root / rebuild.ANCHOR),
            "opponent_checkpoint": str(checkpoint),
            "stochastic_actions": True,
            "stochastic_locomotion_actions": motor,
            "initial_heading_noise_radians": heading,
            "initial_separation": 1.5,
            "duration": 10.0,
            "episode_duration": 10.0,
            "seed": 7000,
            "seeds": 32,
        },
        "scenarios": [
            {"role": role, "rollouts": rows} for role in ("pursuer", "evader")
        ],
    }


class ReplayTests(unittest.TestCase):
    def test_collection_counts_canonical_id_complete_and_missing_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            setup_references(root)
            directory = root / "calibration"
            directory.mkdir()
            for number in (5, 10, 15):
                (directory / f"historical{number}.json").write_text(
                    json.dumps(calibration_file(root, number))
                )
            collection = rebuild.build_collection(directory, root)
            self.assertEqual(collection["protocol"], COLLECTION_PROTOCOL)
            self.assertFalse(collection["groups"][0]["complete"])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                select_calibration(
                    collection,
                    heading_noise_radians=0.0,
                    stochastic_locomotion_actions=True,
                    initial_separation=1.5,
                    duration=10.0,
                )
            (directory / "historical20.json").write_text(
                json.dumps(calibration_file(root, 20))
            )
            group = rebuild.build_collection(directory, root)["groups"][0]
            self.assertTrue(group["complete"])
            opponent_id = rebuild.canonical_historical_id(
                root / rebuild.HISTORICAL / "000000000005"
            )
            self.assertEqual(
                opponent_id,
                "historical/tag_sparse_terminal_640k_seed0/000000000005-000000000000",
            )
            self.assertEqual(
                group["calibration"][opponent_id]["pursuer"],
                {"failures": 32, "episodes": 32},
            )
            self.assertEqual(
                group["calibration"][opponent_id]["evader"],
                {"failures": 0, "episodes": 32},
            )
            self.assertEqual(len(group["sources"]), 4)
            self.assertEqual(len(group["sources"][0]["sha256"]), 64)

    def test_modes_reset_groups_and_protocol_disagreements(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            setup_references(root)
            directory = root / "calibration"
            directory.mkdir()
            for motor in (True, False):
                for number in rebuild.REFERENCES:
                    (directory / f"historical{number}_{motor}.json").write_text(
                        json.dumps(calibration_file(root, number, motor))
                    )
            collection = rebuild.build_collection(directory, root)
            self.assertEqual(len(collection["groups"]), 2)
            self.assertTrue(all(group["complete"] for group in collection["groups"]))
            selected = select_calibration(
                collection,
                heading_noise_radians=0.0,
                stochastic_locomotion_actions=False,
                initial_separation=1.5,
                duration=10.0,
            )
            self.assertIs(selected["stochastic_locomotion_actions"], False)
            bad = calibration_file(root)
            bad["action_mode"] = "deterministic"
            (directory / "historical_bad_mode.json").write_text(json.dumps(bad))
            self.assertTrue(
                any(
                    "sampling" in source["error"]
                    for source in rebuild.build_collection(directory, root)[
                        "rejected_sources"
                    ]
                )
            )
            bad = calibration_file(root)
            bad["checkpoint"] = "wrong_anchor"
            (directory / "historical_bad_anchor.json").write_text(json.dumps(bad))
            self.assertTrue(
                any(
                    "anchor" in source["error"]
                    for source in rebuild.build_collection(directory, root)[
                        "rejected_sources"
                    ]
                )
            )
            with self.assertRaises(ValueError):
                select_calibration(
                    collection,
                    heading_noise_radians=0.5,
                    stochastic_locomotion_actions=True,
                    initial_separation=1.5,
                    duration=10.0,
                )
            with self.assertRaises(ValueError):
                select_calibration(
                    {"protocol": "wrong", "groups": []},
                    heading_noise_radians=0.0,
                    stochastic_locomotion_actions=True,
                    initial_separation=1.5,
                    duration=10.0,
                )

    def test_helper_rejects_unmarked_complete_and_nonstochastic_strategy(self):
        group = {
            "heading_noise_radians": 0.0,
            "stochastic_locomotion_actions": True,
            "initial_separation": 1.5,
            "duration": 10.0,
        }
        collection = {"protocol": COLLECTION_PROTOCOL, "groups": [group]}
        kwargs = {
            "heading_noise_radians": 1e-10,
            "stochastic_locomotion_actions": True,
            "initial_separation": 1.5,
            "duration": 10.0,
        }
        with self.assertRaises(ValueError):
            select_calibration(collection, **kwargs)
        group["complete"] = True
        self.assertIs(select_calibration(collection, **kwargs), group)
        group["strategy_stochastic"] = False
        with self.assertRaises(ValueError):
            select_calibration(collection, **kwargs)
        group["strategy_stochastic"] = True
        collection["groups"].append(dict(group))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_calibration(collection, **kwargs)

    def test_balanced_cause_denominators_and_failure_noncompletion(self):
        record = {
            "rollout_final_causes": [4, 6, 2, 6],
            "metrics": {
                "strategy/pursuer/tag_rate": 0.0,
                "termination/evader/time_limit_truncation_rate": 0.5,
            },
        }
        counts, rates = rebuild.balanced_outcomes(record, {"num_environments": 4})
        self.assertEqual(counts, {"pursuer": 2, "evader": 2})
        self.assertEqual(rates, {"pursuer": 0.0, "evader": 0.5})
        record["metrics"]["strategy/pursuer/episodes"] = 1
        with self.assertRaisesRegex(ValueError, "total role episodes"):
            rebuild.balanced_outcomes(record, {"num_environments": 4})
        record["metrics"].pop("strategy/pursuer/episodes")
        record["metrics"]["strategy/pursuer/tag_rate"] = 1.0
        with self.assertRaisesRegex(ValueError, "Raw final causes"):
            rebuild.balanced_outcomes(record, {"num_environments": 4})

    def test_replay_zero_tags_does_not_use_old_elo_or_write_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "training"
            run.mkdir()
            manifest = {
                "experiment": "hierarchical_joint_tag_ppo",
                "config": {
                    "num_environments": 4,
                    "minimum_initial_separation": 1.5,
                    "maximum_initial_separation": 1.5,
                },
                "wandb": {"id": "original-id"},
            }
            (run / "manifest.json").write_text(json.dumps(manifest))
            record = {
                "metrics_finite": True,
                "environment_steps": 20,
                "opponent_id": "known",
                "rollout_final_causes": [6, 6, 6, 6],
                "metrics": {
                    "strategy/pursuer/tag_rate": 0.0,
                    "termination/evader/time_limit_truncation_rate": 1.0,
                    "elo/combined/perceived_skill": 0.5,
                    "strategy/pursuer/episode_return_mean": -0.1,
                    "strategy/evader/episode_return_mean": 0.1,
                },
            }
            original = json.dumps(record) + "\n"
            (run / "metrics.jsonl").write_text(original)
            group = {
                "calibration": {
                    "known": {
                        role: {"failures": 16, "episodes": 32}
                        for role in ("pursuer", "evader")
                    }
                }
            }
            history, metadata = rebuild.replay_history(run, group, root / "reanalysis")
            self.assertEqual((run / "metrics.jsonl").read_text(), original)
            self.assertNotIn("elo/combined/perceived_skill", history[0])
            self.assertLess(history[0]["benchmark/pursuer/relative_skill"], 0.5)
            self.assertGreater(history[0]["benchmark/evader/relative_skill"], 0.5)
            self.assertEqual(metadata["source_wandb"]["id"], "original-id")


if __name__ == "__main__":
    unittest.main()
