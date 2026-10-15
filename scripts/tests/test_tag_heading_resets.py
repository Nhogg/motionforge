"""Pure CPU reset-noise regressions; no running experiment defaults changed."""

import hashlib
import math
import unittest

import jax
import jax.numpy as jp
import numpy as np

from motionforge.envs.tag_reset import (
    AgentResetLayout,
    TagResetConfig,
    TagResetLayout,
    sample_tag_reset,
)
from motionforge.envs.warp import WarpEnvConfig


class HeadingResetTest(unittest.TestCase):
    def setUp(self):
        self.layout = TagResetLayout(
            (AgentResetLayout(0, slice(7, 36)), AgentResetLayout(36, slice(43, 72)))
        )
        self.keys = jax.random.split(jax.random.PRNGKey(19), 32)

    def sample(self, noise):
        config = TagResetConfig(separation=1.5, heading_noise_radians=noise)
        return jax.jit(
            jax.vmap(
                lambda key: sample_tag_reset(
                    jp.zeros(72), jp.zeros(70), key, self.layout, config
                )
            )
        )(self.keys)

    def test_default_matches_prechange_cpu_fixture(self):
        result = self.sample(0.0)
        # Captured before the optional heading-noise change in this environment.
        digest = hashlib.sha256(np.asarray(result.qpos).tobytes()).hexdigest()
        self.assertEqual(
            digest, "60188d83731591722080e274ede6026b471616cbcd63d8567d03c11f245c8afb"
        )

    def test_noise_reproducible_and_preserves_spawn_and_joints(self):
        baseline = self.sample(0.0)
        noisy = self.sample(math.pi / 2)
        repeat = self.sample(math.pi / 2)
        np.testing.assert_array_equal(noisy.qpos, repeat.qpos)
        np.testing.assert_array_equal(noisy.planar_positions, baseline.planar_positions)
        np.testing.assert_array_equal(noisy.qvel, baseline.qvel)
        np.testing.assert_array_equal(noisy.floor_quaternion, baseline.floor_quaternion)
        delta = np.asarray(noisy.headings - baseline.headings)
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        self.assertLessEqual(np.max(np.abs(delta)), math.pi / 2 + 1e-6)
        self.assertGreater(np.min(np.std(delta, axis=0)), 0.5)
        for agent in self.layout.agents:
            np.testing.assert_array_equal(
                noisy.qpos[:, agent.joint_qpos_slice],
                baseline.qpos[:, agent.joint_qpos_slice],
            )
            q = np.asarray(
                noisy.qpos[:, agent.root_qpos_start + 3 : agent.root_qpos_start + 7]
            )
            np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-6)

    def test_validation_and_warp_config(self):
        for value in (-0.1, math.pi + 0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TagResetConfig(heading_noise_radians=value)
            with self.assertRaises(ValueError):
                WarpEnvConfig(heading_noise_radians=value)
        self.assertEqual(WarpEnvConfig().heading_noise_radians, 0.0)


if __name__ == "__main__":
    unittest.main()
