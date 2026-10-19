"""One first-episode record per physical world, plus evaluation-only TAG recovery."""

import math

import torch


class TagTrials:
  """Freeze the game result at TAG/timeout/failure; keep physics for post-TAG audit.

  The environment must disable TAG auto-termination and auto-reset during this
  evaluation. Recovery cannot turn a recorded TAG into a different game outcome.
  It is a separate quality measurement, not a changed training reward.
  """

  def __init__(self, n, device, dt, episode_s, recovery_s, initial_distance=None):
    self.dt, self.episode_s, self.recovery_s = dt, episode_s, recovery_s
    self.cause = torch.zeros(n, dtype=torch.long, device=device)  # 1 TAG, 2 timeout, 3 failure.
    self.end_step = torch.zeros_like(self.cause)
    self.finished = torch.zeros(n, dtype=torch.bool, device=device)
    self.post_failure = self.finished.clone()
    self.post_flags = {}
    self.minimum = torch.full((n,), float("inf"), device=device)
    self.initial = None if initial_distance is None else initial_distance.clone()
    if self.initial is not None:
      self.minimum = self.initial.clone()
    self.at_end, self.records = {}, []

  def update(self, *, step, roles, tag, failure, physics_done, quality, fallen, oob, failure_flags=None):
    live = self.cause == 0
    self.minimum = torch.where(live, torch.minimum(self.minimum, quality["distance_m"]), self.minimum)
    timeout = step * self.dt >= self.episode_s - 1e-8
    end = live & (tag | failure | timeout)
    cause = torch.where(failure, 3, torch.where(tag, 1, 2))
    self.cause[end], self.end_step[end] = cause[end], step
    flags = failure_flags or {}
    values = {**quality, **flags, "role": roles, "fall": fallen, "oob": oob}
    for name, value in values.items():
      if name not in self.at_end:
        self.at_end[name] = torch.zeros_like(value)
      self.at_end[name][end] = value[end]
    pending = (self.cause == 1) & ~self.finished
    self.post_failure |= pending & failure
    for name, flag in flags.items():
      if name not in self.post_flags:
        self.post_flags[name] = torch.zeros_like(flag)
      self.post_flags[name] |= pending & flag
    recovered = (step - self.end_step) * self.dt >= self.recovery_s - 1e-8
    ready = ~self.finished & ((self.cause > 1) | (pending & (recovered | physics_done)))
    for i in ready.nonzero().flatten().tolist():
      why = int(self.cause[i])
      row = {name: value[i].item() for name, value in self.at_end.items()}
      row.update(
        env_id=i,
        terminal_cause={1: "TAG", 2: "TIMEOUT", 3: "PHYSICAL_FAILURE"}[why],
        tag=why == 1,
        game_duration_s=int(self.end_step[i]) * self.dt,
        minimum_distance_m=self.minimum[i].item(),
        initial_distance_m=self.initial[i].item() if self.initial is not None else None,
        success=(why == 1 and row["role"] > 0) or (why == 2 and row["role"] < 0),
        post_tag_observed_s=(step - int(self.end_step[i])) * self.dt if why == 1 else None,
        post_tag_survived=bool(recovered[i] and not self.post_failure[i]) if why == 1 else None,
      )
      if flags:
        # Stop at the first audit-ending failure: later failures are censored,
        # not evidence the other robot recovered. Keep simultaneous causes and
        # observed OOB even when the separate fall-only audit ignores it.
        causes = lambda source: [
          f"{'pursuer' if (name.startswith('robot_') == (row['role'] > 0)) else 'evader'}_{name.rsplit('_', 1)[1]}"
          for name, value in source.items() if bool(value[i])
        ]
        row["terminal_failure_causes"] = causes({k: self.at_end[k] for k in flags})
        row["post_tag_failure_causes"] = causes(self.post_flags) if why == 1 else None
      self.records.append(row)
    self.finished |= ready
    return ready


def summarize_trials(rows):
  """All first episodes remain in the denominator; failures are never wins."""
  result = {"episodes": len(rows)}
  for name, sign in (("pursuer", 1), ("evader", -1)):
    part = [r for r in rows if r["role"] == sign]
    mean = lambda key: sum(r[key] for r in part) / len(part) if part else None
    result[f"episodes_{name}"] = len(part)
    result[f"win_rate_{name}"] = mean("success")
    result[f"tag_rate_{name}"] = mean("tag")
    if part:
      p, n, z = mean("success"), len(part), 1.959963984540054
      center = (p + z * z / (2 * n)) / (1 + z * z / n)
      half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
      result[f"win_rate_{name}_wilson95"] = [
        0.0 if p == 0 else max(0.0, center - half),
        1.0 if p == 1 else min(1.0, center + half),
      ]
    result[f"physical_failure_rate_{name}"] = (
      sum(r["terminal_cause"] == "PHYSICAL_FAILURE" for r in part) / len(part) if part else None
    )
  tags = [r for r in rows if r["tag"]]
  result["post_tag_survival_rate"] = sum(r["post_tag_survived"] for r in tags) / len(tags) if tags else None
  result["tag_count"] = len(tags)
  if rows and "terminal_failure_causes" in rows[0]:
    for name in ("pursuer", "evader"):
      for cause in ("fall", "oob"):
        key = f"{name}_{cause}"
        result[f"terminal_{key}_rate"] = sum(key in r["terminal_failure_causes"] for r in rows) / len(rows)
        result[f"post_tag_first_failure_{key}_rate"] = (
          sum(key in r["post_tag_failure_causes"] for r in tags) / len(tags) if tags else None
        )
  pursuers = [r for r in rows if r["role"] > 0]
  result["controlled_tag_rate_pursuer"] = (
    sum(r["tag"] and r["post_tag_survived"] for r in pursuers) / len(pursuers) if pursuers else None
  )
  for key in ("closing_speed_mps", "pursuer_root_tilt_deg", "pursuer_angular_speed_radps", "interagent_peak_force_N"):
    result[f"tag_mean_{key}"] = sum(r[key] for r in tags) / len(tags) if tags else None
  result["win_rate"] = sum(r["success"] for r in rows) / len(rows)
  result["fall_rate"] = sum(r["fall"] for r in rows) / len(rows)
  result["mean_episode_length_s"] = sum(r["game_duration_s"] for r in rows) / len(rows)
  return result
