"""High-level policies that choose commands for MotionForge controllers."""

from motionforge.policies.tag_scripted import (
    ScriptedEvaderConfig,
    ScriptedPursuerConfig,
    scripted_evader_command,
    scripted_pursuer_command,
)

__all__ = [
    "ScriptedEvaderConfig",
    "ScriptedPursuerConfig",
    "scripted_evader_command",
    "scripted_pursuer_command",
]
