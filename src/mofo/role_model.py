"""Independent pursuit/evasion networks behind the existing PPO model interface."""

import copy

import torch
from rsl_rl.models import MLPModel
from torch.distributions import Normal


class RoleModel(MLPModel):
  """Route on the raw role input, including independent normalization and noise.

  The inherited MLP is the pursuer; ``evader`` is a complete second model.
  Rewards, action space, observations and PPO losses are otherwise unchanged.
  """

  def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 256, 256), *, role_column, **kwargs):
    kwargs["hidden_dims"] = hidden_dims
    second = copy.deepcopy(kwargs)  # MLPModel consumes distribution_cfg.class_name.
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.evader = MLPModel(obs, obs_groups, obs_set, output_dim, **second)
    self.register_buffer("_role_column", torch.tensor(role_column, dtype=torch.long))
    self.role_column = int(role_column)
    if not 0 <= self.role_column < self.obs_dim:
      raise ValueError("Role column must be inside the unnormalized observation")

  def _mask(self, obs):
    raw = torch.cat([obs[k] for k in self.obs_groups], dim=-1)
    return raw[:, self.role_column:self.role_column + 1] > 0

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
    if masks is not None or hidden_state is not None:
      raise ValueError("RoleModel supports unpadded feedforward observations only")
    self.pursuer_mask = self._mask(obs)
    p = super().forward(obs, stochastic_output=stochastic_output)
    e = self.evader(obs, stochastic_output=stochastic_output)
    return torch.where(self.pursuer_mask, p, e)

  @property
  def output_mean(self):
    return torch.where(self.pursuer_mask, self.distribution.mean, self.evader.output_mean)

  @property
  def output_std(self):
    return torch.where(self.pursuer_mask, self.distribution.std, self.evader.output_std)

  @property
  def output_entropy(self):
    return torch.where(self.pursuer_mask[:, 0], self.distribution.entropy, self.evader.output_entropy)

  @property
  def output_distribution_params(self):
    return self.output_mean, self.output_std

  def get_output_log_prob(self, outputs):
    return Normal(self.output_mean, self.output_std).log_prob(outputs).sum(-1)

  def update_normalization(self, obs):
    p = self._mask(obs)[:, 0]
    if p.any():
      super().update_normalization(obs[p])
    if (~p).any():
      self.evader.update_normalization(obs[~p])

  def load_shared_state_dict(self, state):
    """Exact initialization of both roles from an existing shared TAG model."""
    self.load_state_dict({**state, **{f"evader.{k}": v for k, v in state.items()},
                          "_role_column": self._role_column})

  def as_jit(self):
    return RolePolicy(super().as_jit(), self.evader.as_jit(), self.role_column)

  def as_onnx(self, verbose=False):
    raise NotImplementedError("Use the role-aware TorchScript export")


class RolePolicy(torch.nn.Module):
  """Deterministic checkpoint/snapshot inference with the same raw role routing."""

  def __init__(self, pursuer, evader, role_column):
    super().__init__()
    self.pursuer, self.evader, self.role_column = pursuer, evader, role_column

  def forward(self, obs):
    return torch.where(obs[:, self.role_column:self.role_column + 1] > 0,
                       self.pursuer(obs), self.evader(obs))
