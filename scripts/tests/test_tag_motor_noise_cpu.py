"""Pure NumPy pairing/formula tests; no GPU or simulator initialization."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location(
    "audit_tag_motor_noise_cpu", DIRECTORY / "audit_tag_motor_noise_cpu.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class MotorNoiseTests(unittest.TestCase):
    def test_common_noise_reproducible(self):
        np.testing.assert_array_equal(audit.paired_noise(101), audit.paired_noise(101))
        self.assertEqual(audit.paired_noise(101).shape, (125, 2, 29))
        self.assertFalse(
            np.array_equal(audit.paired_noise(101), audit.paired_noise(102))
        )

    def test_formulas_and_zero_noise(self):
        location = np.array([-0.5, 0.2, 2.0], dtype=np.float32)
        innovation = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        migrated = np.array([0.1, 0.2, 0.3])
        original = np.array([0.3, 0.2, 0.1])
        np.testing.assert_allclose(
            audit.motor_action(
                location, migrated, original, innovation, "deterministic"
            ),
            np.tanh(location),
        )
        np.testing.assert_allclose(
            audit.motor_action(
                location, migrated, original, innovation, "migrated_gaussian"
            ),
            np.tanh(location + migrated * innovation),
        )
        np.testing.assert_allclose(
            audit.motor_action(
                location, migrated, original, innovation, "original_p2_gaussian"
            ),
            np.tanh(location + original * innovation),
        )
        for mode in audit.MODES:
            np.testing.assert_allclose(
                audit.motor_action(location, migrated, original, np.zeros(3), mode),
                np.tanh(location),
            )
        with self.assertRaises(ValueError):
            audit.motor_action(location, migrated, original, innovation, "unknown")

    def test_measurement_window_excludes_warmup(self):
        trace = [
            {"measurement_window": False, "body_vx_vy_gyro_yaw": [[100, 100, 100]] * 2},
            {"measurement_window": True, "body_vx_vy_gyro_yaw": [[0.8, 0, 1]] * 2},
        ]
        result = audit.summarize_trace(trace, (0.8, 0, 1))
        self.assertEqual(result["measurement_samples"], 1)
        np.testing.assert_allclose(result["tracking_rmse_vx_vy_yaw_by_agent"], 0)


if __name__ == "__main__":
    unittest.main()
