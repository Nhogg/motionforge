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
    return {
        "agent_roles": names,
        "agent_role_encodings": actual.tolist(),
        "selected_strategy_heads": [f"{name}_head" for name in names],
        "pursuer_index": pursuer_index,
        "evader_index": 1 - pursuer_index,
        "learner_index": segment.learner_index,
        "opponent_index": segment.opponent_index,
        "learner_role": segment.learner_role,
        "opponent_role": segment.opponent_role,
        "colors": {"green": "pursuer", "orange": "evader"},
        "roles_validated": True,
    }


@lru_cache(maxsize=8)
def _font(size):
    return ImageFont.load_default(size=size)


def annotate_hierarchical_frame(frame, audit, *, stochastic_actions, matchup):
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
    draw.text(
        (6, 2 * size + 9),
        f"Learner: {audit['learner_role']} | {matchup} | {mode}",
        font=font,
        fill=(245, 245, 245),
    )
    return np.asarray(result).copy()
