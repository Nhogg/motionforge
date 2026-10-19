"""PPO whose KL-adaptive learning rate cannot exceed a configured ceiling."""

from rsl_rl.algorithms import PPO


class CappedPPO(PPO):
  """rsl-rl raises the adaptive rate up to 1e-2; long TAG runs destabilize well below that."""

  def __init__(self, *args, max_learning_rate: float, **kwargs):
    self.max_learning_rate = max_learning_rate
    super().__init__(*args, **kwargs)

  @property
  def learning_rate(self) -> float:
    return self._learning_rate

  @learning_rate.setter
  def learning_rate(self, value: float) -> None:
    # Every assignment, including the adaptive update, is clamped before the optimizer sees it.
    self._learning_rate = min(value, self.max_learning_rate)
