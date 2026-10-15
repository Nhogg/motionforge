"""Host-side command/frame diagnostics independent of game rewards and ratings."""

import numpy as np


def motion_audit(
    positions, quaternions, commands, raw_commands, relative_positions, dt
):
    positions, q, commands, raw, relative = map(
        np.asarray, (positions, quaternions, commands, raw_commands, relative_positions)
    )
    if dt <= 0 or len(positions) < 2:
        raise ValueError("Need positive dt and at least two samples")
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    delta = positions[:, ::-1] - positions
    expected_relative = np.stack(
        (c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]),
        axis=-1,
    )
    velocity = np.diff(positions, axis=0) / dt
    # Command at sample t+1 drove the interval from physical state t to t+1.
    cmd = commands[1:]
    world_command = np.stack(
        (
            c[:-1] * cmd[..., 0] - s[:-1] * cmd[..., 1],
            s[:-1] * cmd[..., 0] + c[:-1] * cmd[..., 1],
        ),
        axis=-1,
    )
    body_velocity = np.stack(
        (
            c[:-1] * velocity[..., 0] + s[:-1] * velocity[..., 1],
            -s[:-1] * velocity[..., 0] + c[:-1] * velocity[..., 1],
        ),
        axis=-1,
    )
    yaw_rate = np.diff(np.unwrap(yaw, axis=0), axis=0) / dt
    results = {
        "max_relative_position_frame_error_m": float(
            np.max(np.abs(expected_relative - relative))
        ),
        "agents": [],
    }
    for a in range(2):
        path = float(np.linalg.norm(np.diff(positions[:, a], axis=0), axis=-1).sum())
        to_target = delta[:-1, a] / np.maximum(
            np.linalg.norm(delta[:-1, a], axis=-1, keepdims=True), 1e-8
        )
        results["agents"].append(
            {
                "agent_index": a,
                "path_length_m": path,
                "net_displacement_m": float(
                    np.linalg.norm(positions[-1, a] - positions[0, a])
                ),
                "path_efficiency": float(
                    np.linalg.norm(positions[-1, a] - positions[0, a]) / max(path, 1e-8)
                ),
                "signed_heading_turns": float(
                    np.sum(yaw_rate[:, a]) * dt / (2 * np.pi)
                ),
                "absolute_heading_turns": float(
                    np.sum(np.abs(yaw_rate[:, a])) * dt / (2 * np.pi)
                ),
                "backward_motion_fraction": float(
                    np.mean(body_velocity[:, a, 0] < -0.05)
                ),
                "backward_command_fraction": float(np.mean(cmd[:, a, 0] < -0.05)),
                "mean_command_yaw": float(np.mean(cmd[:, a, 2])),
                "mean_actual_yaw_rate": float(np.mean(yaw_rate[:, a])),
                "world_velocity_tracking_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.sum((velocity[:, a] - world_command[:, a]) ** 2, axis=-1)
                        )
                    )
                ),
                "yaw_tracking_rmse": float(
                    np.sqrt(np.mean((yaw_rate[:, a] - cmd[:, a, 2]) ** 2))
                ),
                "mean_command_toward_opponent_mps": float(
                    np.mean(np.sum(world_command[:, a] * to_target, axis=-1))
                ),
                "mean_actual_toward_opponent_mps": float(
                    np.mean(np.sum(velocity[:, a] * to_target, axis=-1))
                ),
                "command_intervention_rms": float(
                    np.sqrt(np.mean((raw[1:, a] - cmd[:, a]) ** 2))
                ),
            }
        )
    return results
