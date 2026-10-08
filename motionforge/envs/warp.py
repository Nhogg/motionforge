"""Low-level two-G1 MJX-Warp simulation without task or league semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jp
from flax import struct
from mujoco import mjx

from motionforge.envs.tag_reset import (
    TagResetConfig,
    build_tag_reset_layout,
    sample_tag_reset,
)
from motionforge.envs.two_g1 import TwoG1Model, build_two_g1_model, make_two_g1_data


@dataclass(frozen=True)
class WarpEnvConfig:
    simulation_timestep: float = 0.002
    control_timestep: float = 0.02
    separation: float = 2.0
    slope_degrees: float = 0.0
    slope_curriculum_degrees: float = 0.0
    naconmax: int = 64
    njmax: int = 256

    def __post_init__(self) -> None:
        if self.simulation_timestep <= 0.0:
            raise ValueError("simulation_timestep must be positive")
        if self.control_timestep <= 0.0:
            raise ValueError("control_timestep must be positive")
        substeps = self.control_timestep / self.simulation_timestep
        if not math.isclose(substeps, round(substeps), abs_tol=1e-9):
            raise ValueError(
                "control_timestep must be an integer multiple of "
                "simulation_timestep"
            )
        if self.naconmax <= 0:
            raise ValueError("naconmax must be positive")
        if self.njmax <= 0:
            raise ValueError("njmax must be positive")
        TagResetConfig(
            separation=self.separation,
            slope_degrees=self.slope_degrees,
            slope_curriculum_degrees=self.slope_curriculum_degrees,
        )

    @property
    def physics_substeps(self) -> int:
        return round(self.control_timestep / self.simulation_timestep)


@struct.dataclass
class WarpEnvState:
    """JAX-compatible state owned solely by the low-level simulator."""

    data: mjx.Data
    rng: jax.Array
    step_count: jax.Array


class WarpEnv:
    """Two-agent Warp simulation accepting 29 joint targets per robot."""

    def __init__(self, config: WarpEnvConfig | None = None) -> None:
        self.config = WarpEnvConfig() if config is None else config
        self.model_bundle: TwoG1Model = build_two_g1_model(
            timestep=self.config.simulation_timestep,
            enable_inter_agent_collision=True,
            slope_degrees=self.config.slope_degrees,
        )
        native_data = make_two_g1_data(
            self.model_bundle,
            separation=self.config.separation,
        )
        self.model = mjx.put_model(self.model_bundle.model, impl="warp")
        self.template_data = mjx.put_data(
            self.model_bundle.model,
            native_data,
            impl="warp",
            naconmax=self.config.naconmax,
            njmax=self.config.njmax,
        )
        self.reset_layout = build_tag_reset_layout(self.model_bundle)
        self.reset_config = TagResetConfig(
            separation=self.config.separation,
            slope_degrees=self.config.slope_degrees,
            slope_curriculum_degrees=self.config.slope_curriculum_degrees,
        )
        self.actuator_ids = jp.asarray(
            [agent.actuator_ids for agent in self.model_bundle.agents],
            dtype=jp.int32,
        )
        self.default_joint_targets = self.template_data.ctrl[self.actuator_ids]

    @property
    def action_shape(self) -> tuple[int, int]:
        return (2, 29)

    def reset(self, key: jax.Array) -> WarpEnvState:
        """Reset physical state without assigning roles or task outcomes."""
        reset_key, next_key = jax.random.split(key)
        sampled = sample_tag_reset(
            self.template_data.qpos,
            self.template_data.qvel,
            reset_key,
            self.reset_layout,
            self.reset_config,
        )
        data = self.template_data.replace(
            qpos=sampled.qpos,
            qvel=sampled.qvel,
            mocap_quat=self.template_data.mocap_quat.at[
                self.model_bundle.terrain_mocap_id
            ].set(sampled.floor_quaternion),
        )
        data = mjx.forward(self.model, data)
        return WarpEnvState(
            data=data,
            rng=next_key,
            step_count=jp.zeros((), dtype=jp.int32),
        )

    def step(
        self,
        state: WarpEnvState,
        joint_position_targets: jax.Array,
    ) -> WarpEnvState:
        """Advance one 50 Hz control update through Warp physics."""
        if joint_position_targets.shape != self.action_shape:
            raise ValueError(
                "joint_position_targets must have shape "
                f"{self.action_shape}; got {joint_position_targets.shape}"
            )
        controls = state.data.ctrl.at[self.actuator_ids].set(joint_position_targets)
        data = state.data.replace(ctrl=controls)

        def physics_step(current_data, _):
            return mjx.step(self.model, current_data), None

        data, _ = jax.lax.scan(
            physics_step,
            data,
            xs=None,
            length=self.config.physics_substeps,
        )
        return state.replace(data=data, step_count=state.step_count + 1)
