"""NoOp Starts wrapper to allow stochastic initial state for deterministic Python environments."""

from typing import Optional

from acme.wrappers import base
import dm_env
import numpy as np

class NoopStartsWrapper(base.EnvironmentWrapper):
  """Implements random noop starts to episodes."""

  def __init__(self,
               environment: dm_env.Environment,
               noop_action: int = 0,
               noop_max: int = 30,
               seed: Optional[int] = None):
    """Initializes a `_NoopReset` wrapper.

    Args:
      environment: An Atari environment.
      noop_action: The noop action used to step the environment for random initialisation.
      noop_max: The maximal number of noop actions at the start of an episode.
      seed: The random seed used to sample the number of noops.
    """
    if noop_max < 0:
      raise ValueError(
        "Maximal number of no-ops after reset cannot be negative. "
        f"Received noop_max={noop_max}"
      )

    super().__init__(environment)
    self.np_random = np.random.RandomState(seed)
    self._noop_max = noop_max
    self._noop_action = noop_action

  def reset(self) -> dm_env.TimeStep:
    """Resets environment and provides the first timestep."""
    noops = (
      self.np_random.randint(self._noop_max + 1)
      if self._noop_max > 0
      else 0
    )
    timestep = self.environment.reset()
    for _ in range(noops):
      timestep = self.environment.step(self._noop_action)
      if timestep.last():
        timestep = self.environment.reset()

    return timestep._replace(step_type=dm_env.StepType.FIRST)
