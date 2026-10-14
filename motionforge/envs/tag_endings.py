"""Exclusive TAG ending classification; time limits censor the game outcome."""

from enum import IntEnum

import jax.numpy as jp


class TagEndCause(IntEnum):
    NONE = 0
    TAG = 1
    PURSUER_FALL = 2
    EVADER_FALL = 3
    PURSUER_OOB = 4
    EVADER_OOB = 5
    TIME_LIMIT_TRUNCATION = 6
    NONFINITE = 7


def classify_tag_ending(
    *, contact, fallen, out_of_bounds, nonfinite, time_limit, pursuer_index=0
):
    """Failure beats contact, true termination beats timeout; role 0 wins ties.

    Both offending agents retain their failure flags/penalties when simultaneous
    failures occur, but the primary cause is unique. Nonfinite has top priority.
    """
    p, e = pursuer_index, 1 - pursuer_index
    failure = jp.any(fallen) | jp.any(out_of_bounds) | nonfinite
    tagged = contact & ~failure
    terminated = failure | tagged
    truncated = time_limit & ~terminated
    cause = jp.where(truncated, TagEndCause.TIME_LIMIT_TRUNCATION, TagEndCause.NONE)
    cause = jp.where(tagged, TagEndCause.TAG, cause)
    for flag, code in (
        (out_of_bounds[e], TagEndCause.EVADER_OOB),
        (out_of_bounds[p], TagEndCause.PURSUER_OOB),
        (fallen[e], TagEndCause.EVADER_FALL),
        (fallen[p], TagEndCause.PURSUER_FALL),
        (nonfinite, TagEndCause.NONFINITE),
    ):
        cause = jp.where(flag, code, cause)
    return tagged, terminated, truncated, cause.astype(jp.int32)


def tag_ending_metadata(ending, *, collection_cutoff=False):
    """Host-side validation/serialization for videos and evaluation artifacts."""
    terminated, truncated = bool(ending.terminated), bool(ending.truncated)
    cause = TagEndCause(int(ending.cause))
    if collection_cutoff and cause == TagEndCause.NONE:
        truncated, cause = True, TagEndCause.TIME_LIMIT_TRUNCATION
    if terminated and truncated:
        raise AssertionError("TAG transition cannot be terminated and truncated")
    if (terminated or truncated) != (cause != TagEndCause.NONE):
        raise AssertionError("TAG ending flags and cause disagree")
    return {
        "terminated": terminated,
        "truncated": truncated,
        "terminal_cause": cause.name,
        "nonfinite": bool(ending.nonfinite),
    }
