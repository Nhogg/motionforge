"""Synthetic regressions for heading frames and backward-command diagnostics."""

import numpy as np

from motionforge.evaluation.tag_motion_audit import motion_audit


def main():
    for yaw in (0.0, np.pi / 2, -np.pi / 2, np.pi):
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        for speed in (0.5, -0.5):
            positions = np.zeros((11, 2, 2))
            positions[:, 0] = np.arange(11)[:, None] * 0.02 * speed * heading
            positions[:, 1] = positions[:, 0] + np.array([2.0, 1.0])
            q = np.broadcast_to([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], (11, 2, 4))
            commands = np.broadcast_to([speed, 0, 0], (11, 2, 3))
            delta = positions[:, ::-1] - positions
            c, s = np.cos(yaw), np.sin(yaw)
            relative = np.stack(
                (
                    c * delta[..., 0] + s * delta[..., 1],
                    -s * delta[..., 0] + c * delta[..., 1],
                ),
                axis=-1,
            )
            result = motion_audit(positions, q, commands, commands, relative, 0.02)
            assert result["max_relative_position_frame_error_m"] < 1e-6
            for a in result["agents"]:
                assert a["world_velocity_tracking_rmse"] < 1e-6
                assert a["backward_motion_fraction"] == float(speed < 0)
                assert a["backward_command_fraction"] == float(speed < 0)
                assert abs(a["path_efficiency"] - 1) < 1e-6
    print("Motion audit frame/sign/alignment checks passed")


if __name__ == "__main__":
    main()
