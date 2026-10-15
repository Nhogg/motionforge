"""CPU fixtures for the checkpoint-content freeze audit."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from flax import serialization

SPEC = importlib.util.spec_from_file_location(
    "audit_tag_frozen_checkpoints",
    Path(__file__).resolve().parents[1]
    / "evaluation/audit_tag_frozen_checkpoints.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class FrozenCheckpointAuditTests(unittest.TestCase):
    def test_order_independent_but_sensitive_to_parameters_and_optimizer(self):
        first = {"weight": np.array([1, 2], dtype=np.float32), "count": 0}
        reordered = {"count": 0, "weight": first["weight"].copy()}
        self.assertEqual(audit.digest_tree(first), audit.digest_tree(reordered))
        for changed in (
            {**first, "weight": np.array([1, 3], dtype=np.float32)},
            {**first, "weight": first["weight"].astype(np.float64)},
            {**first, "weight": first["weight"].reshape(1, 2)},
            {**first, "count": 1},
        ):
            self.assertNotEqual(audit.digest_tree(first), audit.digest_tree(changed))

    def test_reads_actual_contents_and_rejects_payload_corruption(self):
        state = {
            "parameters": {"locomotion": {"weight": np.ones(3, np.float32)}},
            "locomotion_optimizer_state": {"count": np.array(0, np.int32)},
            "locomotion_updates": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            payload = serialization.msgpack_serialize(state)
            path = checkpoint / "learner_state.msgpack"
            path.write_bytes(payload)
            (checkpoint / "manifest.json").write_text(
                json.dumps({"state_sha256": hashlib.sha256(payload).hexdigest()})
            )
            actual = audit.read_state(checkpoint)
            self.assertEqual(actual["locomotion_updates"], 0)
            self.assertEqual(
                actual["motor_parameter_sha256"],
                audit.digest_tree(state["parameters"]["locomotion"]),
            )
            self.assertEqual(
                actual["motor_optimizer_sha256"],
                audit.digest_tree(state["locomotion_optimizer_state"]),
            )
            path.write_bytes(payload + b"corruption")
            with self.assertRaisesRegex(ValueError, "Checkpoint digest mismatch"):
                audit.read_state(checkpoint)

    def test_object_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected object array"):
            audit.digest_tree(np.array([object()], dtype=object))


if __name__ == "__main__":
    unittest.main()
