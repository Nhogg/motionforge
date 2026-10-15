"""Validated physical-agent roles and consistent labels for TAG diagnostics."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PURSUER_COLOR = (0.20, 0.75, 0.30)
EVADER_COLOR = (0.90, 0.35, 0.15)


def color_hierarchical_agents(model_bundle, *, pursuer_index):
    """Color visible meshes too; collision-only geoms do not label a robot."""
    model = model_bundle.model
    counts = [0, 0]
    for geom_id in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom_id])).name or ""
        for index, agent in enumerate(model_bundle.agents):
            if body_name.startswith(agent.prefix):
                model.geom_rgba[geom_id, :3] = (
                    PURSUER_COLOR if index == pursuer_index else EVADER_COLOR
                )
                # Native-model rendering only: compiled Warp physics is unchanged.
                model.geom_matid[geom_id] = -1
                counts[index] += 1
                break
    if not all(counts):
        raise ValueError("could not resolve both agents' render geoms")
    return counts


def audit_hierarchical_roles(*, pursuer_index, segment, roles) -> dict:
    if pursuer_index not in (0, 1):
        raise ValueError("pursuer_index must be zero or one")
    expected = np.eye(2, dtype=np.float32)
    if pursuer_index == 1:
        expected = expected[::-1]
    actual = np.asarray(roles)
    if actual.shape != (2, 2) or not np.array_equal(actual, expected):
        raise ValueError(f"incorrect physical-agent roles: {actual.tolist()}")
    expected_learner = (
        pursuer_index if segment.learner_is_pursuer else 1 - pursuer_index
    )
    if segment.learner_index != expected_learner:
        raise ValueError("learner role disagrees with its physical-agent index")
    names = ["pursuer" if i == pursuer_index else "evader" for i in range(2)]
    bindings = [
        {
            "physical_agent_index": i,
            "role": name,
            "strategy_head": f"{name}_head",
            "parameter_source": (
                "learner" if i == segment.learner_index else "opponent"
            ),
        }
        for i, name in enumerate(names)
    ]
    return {
        "agent_roles": names,
        "agent_role_encodings": actual.tolist(),
        "selected_strategy_heads": [f"{name}_head" for name in names],
        "agent_policy_bindings": bindings,
        "pursuer_index": pursuer_index,
        "evader_index": 1 - pursuer_index,
        "learner_index": segment.learner_index,
        "opponent_index": segment.opponent_index,
        "learner_role": segment.learner_role,
        "opponent_role": segment.opponent_role,
        "colors": {"green": "pursuer", "orange": "evader"},
        "roles_validated": True,
    }


def audit_hierarchical_role_trace(role_trace, *, pursuer_index) -> dict:
    """Validate the physical role encoding at every collected rollout sample."""
    if pursuer_index not in (0, 1):
        raise ValueError("pursuer_index must be zero or one")
    expected = np.eye(2, dtype=np.float32)
    if pursuer_index == 1:
        expected = expected[::-1]
    actual = np.asarray(role_trace)
    if actual.ndim != 3 or actual.shape[1:] != (2, 2):
        raise ValueError(
            "role trace must have shape (steps, 2, 2); "
            f"got {actual.shape}"
        )
    if not np.all(np.isin(actual, (0.0, 1.0))):
        raise ValueError("role trace contains non-binary encodings")
    if not np.all(np.asarray(actual.sum(axis=-1)) == 1.0):
        raise ValueError("role trace contains non-one-hot agent encodings")
    if not np.all(np.asarray(actual.sum(axis=1)) == 1.0):
        raise ValueError("role trace contains duplicate physical roles")
    if not np.array_equal(actual, np.broadcast_to(expected, actual.shape)):
        raise ValueError("physical-agent role encodings changed during rollout")
    return {
        "agent_role_trace": actual.tolist(),
        "role_trace_steps": int(actual.shape[0]),
        "role_trace_validated": True,
    }


@lru_cache(maxsize=8)
def _font(size):
    return ImageFont.load_default(size=size)


def annotate_hierarchical_frame(
    frame, audit, *, stochastic_actions, matchup, stochastic_locomotion_actions=None
):
    """Put the actual roles in the video itself, independent of its caption."""
    result = Image.fromarray(frame)
    draw = ImageDraw.Draw(result)
    size = max(10, frame.shape[1] // 70)
    font = _font(size)
    draw.rectangle((0, 0, frame.shape[1], size * 3 + 12), fill=(15, 15, 15))
    draw.text(
        (6, 3),
        f"GREEN: pursuer (agent {audit['pursuer_index']})",
        font=font,
        fill=tuple(int(255 * v) for v in PURSUER_COLOR),
    )
    draw.text(
        (6, size + 6),
        f"ORANGE: evader (agent {audit['evader_index']})",
        font=font,
        fill=tuple(int(255 * v) for v in EVADER_COLOR),
    )
    mode = "stochastic" if stochastic_actions else "deterministic"
    if (
        stochastic_locomotion_actions is not None
        and stochastic_locomotion_actions != stochastic_actions
    ):
        motor_mode = "stochastic" if stochastic_locomotion_actions else "deterministic"
        mode = f"strategy {mode}, motor {motor_mode}"
    draw.text(
        (6, 2 * size + 9),
        f"Learner: {audit['learner_role']} | {matchup} | {mode}",
        font=font,
        fill=(245, 245, 245),
    )
    return np.asarray(result).copy()
