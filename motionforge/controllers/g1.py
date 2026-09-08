"""g1.py

Author: Nathan hogg <nathanhogg1223@gmail.com>
Description:
    The public high-level input is desired pelvis-frame forward velocity, lateral
    velocity, and yaw rate. The controller consumes an environment-produced
    policy observation adn returns both the 29 normalized policy actions
    expected by the playground evironment and their corresponding joint-position
    targets. It is stateless apart from immutable checkpoint params.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jp
from brax.training import checkpoint
from brax.training.agents.ppo import networks as ppo_networks

from motionforge.compat.brax_checkpoint import load_ppo_network

COMMAND_OBSERVATION_SLICE = slice(9, 12)


@dataclass(frozen=True)
class G1VelocityCommand:
    """Desired pelvis-frame planar velocity and yaw rate."""

    forward_velocity: float
    lateral_velocity: float
    yaw_rate: float

    def as_array(self) -> jax.Array:
        """Return the policy command ordered as ``[vx, vy, yaw_rate]``."""
        return jp.asarray(
            [self.forward_velocity, self.lateral_velocity, self.yaw_rate],
            dtype=jp.float32,
        )


class G1ControlOutput(NamedTuple):
    """Low-level action and the actuator targets it represents."""

    normalized_action: jax.Array
    joint_position_targets: jax.Array


def apply_velocity_command(state: Any, command: jax.Array) -> Any:
    """Write ``[vx, vy, yaw_rate]`` into state and policy observations."""
    if command.shape != (3,):
        raise ValueError(
            f"G1 velocity command must have shape (3,), got {command.shape}"
        )

    info = dict(state.info)
    info["command"] = command

    observation = dict(state.obs)
    for observation_name in ("state", "privileged_state"):
        observation[observation_name] = (
            observation[observation_name].at[COMMAND_OBSERVATION_SLICE].set(command)
        )

    return state.replace(info=info, obs=observation)


class G1LocomotionController:
    """Stateless adapter from a G1 policy observation to actuator commands."""

    def __init__(
        self,
        *,
        policy: Any,
        default_pose: jax.Array,
        action_scale: float,
        checkpoint_path: Path,
    ) -> None:
        self._policy = policy
        self._default_pose = jp.asarray(default_pose)
        self._action_scale = float(action_scale)
        self._checkpoint_path = checkpoint_path

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        environment: Any,
        *,
        deterministic: bool = True,
    ) -> G1LocomotionController:
        """Restore a Brax PPO checkpoint for the supplied G1 environment."""
        resolved_path = checkpoint_path.resolve()
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"Checkpoint does not exist: {resolved_path}")

        network = load_ppo_network(resolved_path / "ppo_network_config.json")
        parameters = checkpoint.load(resolved_path)
        policy = ppo_networks.make_inference_fn(network)(
            parameters,
            deterministic=deterministic,
        )
        return cls(
            policy=policy,
            default_pose=environment._default_pose,
            action_scale=environment._config.action_scale,
            checkpoint_path=resolved_path,
        )

    @property
    def action_size(self) -> int:
        """Number of normalized actuator actions emitted per robot."""
        return int(self._default_pose.shape[0])

    @property
    def checkpoint_path(self) -> Path:
        """Resolved checkpoint supplying the policy parameters."""
        return self._checkpoint_path

    def act(self, observation: Any, rng: jax.Array) -> G1ControlOutput:
        """Compute one low-level control update without stepping physics."""
        normalized_action, _ = self._policy(observation, rng)
        joint_position_targets = (
            self._default_pose + normalized_action * self._action_scale
        )
        return G1ControlOutput(normalized_action, joint_position_targets)
