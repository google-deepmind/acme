# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rewarder class implementation."""

from typing import Iterator

from acme import types
import jax
import jax.numpy as jnp
import numpy as np


class WassersteinDistanceRewarder:
  """Computes PWIL rewards along a trajectory.

  The rewards measure similarity to the demonstration transitions and are based
  on a greedy approximation to the Wasserstein distance between trajectories.
  """

  def __init__(self,
               demonstrations_it: Iterator[types.Transition],
               episode_length: int,
               use_actions_for_distance: bool = False,
               alpha: float = 5.,
               beta: float = 5.):
    """Initializes the rewarder.

    Args:
      demonstrations_it: An iterator over acme.types.Transition.
      episode_length: a target episode length (policies will be encouraged by
        the imitation reward to have that length).
      use_actions_for_distance: whether to use action to compute reward.
      alpha: float scaling the reward function.
      beta: float controling the kernel size of the reward function.
    """
    self._episode_length = episode_length

    self._use_actions_for_distance = use_actions_for_distance
    self._vectorized_demonstrations = self._vectorize(demonstrations_it)

    # Observations and actions are flat.
    atom_dims = self._vectorized_demonstrations.shape[1]
    self._reward_sigma = beta * self._episode_length / np.sqrt(atom_dims)
    self._reward_scale = alpha

    self._std = np.std(self._vectorized_demonstrations, axis=0, dtype='float64')
    # The std is set to 1 if the observation values are below a threshold.
    # This prevents normalizing observation values that are constant (which can
    # be problematic with e.g. demonstrations coming from a different version
    # of the environment and where the constant values are slightly different).
    self._std = (self._std < 1e-6) + self._std

    self.expert_atoms = self._vectorized_demonstrations / self._std
    self._compute_norm = jax.jit(lambda a, b: jnp.linalg.norm(a - b, axis=1),
                                 device=jax.devices('cpu')[0])

  def _vectorize(self,
                 demonstrations_it: Iterator[types.Transition]) -> np.ndarray:
    """Converts filtered expert demonstrations to numpy array.

    Args:
      demonstrations_it: list of expert demonstrations

    Returns:
      numpy array with dimension:
      [num_expert_transitions, dim_observation] if not use_actions_for_distance
      [num_expert_transitions, (dim_observation + dim_action)] otherwise
    """
    if self._use_actions_for_distance:
      demonstrations = [
          np.concatenate([t.observation, t.action]) for t in demonstrations_it
      ]
    else:
      demonstrations = [t.observation for t in demonstrations_it]
    return np.array(demonstrations)

  def reset(self) -> None:
    """Makes all expert transitions available and initialize weights."""
    num_expert_atoms = len(self.expert_atoms)
    self._all_expert_weights_zero = False
    self.expert_weights = np.ones(num_expert_atoms) / num_expert_atoms

  def append_and_compute_reward(self, observation: jnp.ndarray,
                                action: jnp.ndarray) -> np.float32:
    """Computes reward and updates state, advancing it along a trajectory.

    Subsequent calls to append_and_compute_reward assume inputs are subsequent
    trajectory points.

    Args:
      observation: observation on a trajectory, to compare with the expert
        demonstration(s).
      action: the action following the observation on the trajectory.

    Returns:
      the reward value: the return contribution from the trajectory point.

    """
    # If we run out of demonstrations, penalize further action.
    if self._all_expert_weights_zero:
      return np.float32(0.)

    # Scale observation and action.
    if self._use_actions_for_distance:
      agent_atom = np.concatenate([observation, action])
    else:
      agent_atom = observation
    agent_atom /= self._std

    cost = 0.
    # A special marker for records with zero expert weight. Has to be large so
    # that argmin will not return it.
    DELETED = 1e10  # pylint: disable=invalid-name
    # As we match the expert's weights with the agent's weights, we might
    # raise an error due to float precision, we substract a small epsilon from
    # the agent's weights to prevent that.
    weight = 1. / self._episode_length - 1e-6
    norms = np.array(self._compute_norm(self.expert_atoms, agent_atom))
    # We need to mask out states with zero weight, so that 'argmin' would not
    # return them.
    adjusted_norms = (1 - np.sign(self.expert_weights)) * DELETED + norms
    while weight > 0:
      # Get closest expert state action to agent's state action.
      argmin = adjusted_norms.argmin()
      effective_weight = min(weight, self.expert_weights[argmin])

      if adjusted_norms[argmin] >= DELETED:
        self._all_expert_weights_zero = True
        break

      # Update cost and weights.
      weight -= effective_weight
      self.expert_weights[argmin] -= effective_weight
      cost += effective_weight * norms[argmin]
      adjusted_norms[argmin] = DELETED

    if weight > 0:
      # We have a 'partial' cost if we ran out of demonstrations in the reward
      # computation loop. We assign a high cost (infinite) in this case which
      # makes the reward equal to 0.
      reward = np.array(0.)
    else:
      reward = self._reward_scale * np.exp(-self._reward_sigma * cost)

    return reward.astype('float32')
