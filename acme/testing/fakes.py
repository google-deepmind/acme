# python3
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

"""Fake (mock) components.

Minimal implementations of fake Acme components which can be instantiated in
order to test or interact with other components.
"""

from typing import List, Sequence

from acme import core
from acme import specs
from acme import types

import dm_env
import numpy as np
import tree


class Actor(core.Actor):
  """Fake actor which generates random actions and validates specs."""

  def __init__(self, spec: specs.EnvironmentSpec):
    self._spec = spec
    self.num_updates = 0

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    _validate_spec(self._spec.observations, observation)
    return _generate_from_spec(self._spec.actions)

  def observe_first(self, timestep: dm_env.TimeStep):
    _validate_spec(self._spec.observations, timestep.observation)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    _validate_spec(self._spec.actions, action)
    _validate_spec(self._spec.rewards, next_timestep.reward)
    _validate_spec(self._spec.discounts, next_timestep.discount)
    _validate_spec(self._spec.observations, next_timestep.observation)

  def update(self):
    self.num_updates += 1


class VariableSource(core.VariableSource):
  """Fake variable source."""

  def __init__(self, variables: types.NestedArray = None):
    # Add dummy variables so we can expose them in get_variables.
    self._variables = {'policy': [] if not variables else variables}

  def get_variables(self, names: List[str]) -> List[types.NestedArray]:
    return [self._variables[name] for name in names]


class Environment(dm_env.Environment):
  """A fake environment with a given spec."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      *,
      episode_length: int = 25,
  ):
    # Assert that the discount spec is a BoundedArray with range [0, 1].
    if (not isinstance(spec.discounts, specs.BoundedArray) or
        not np.isclose(spec.discounts.minimum, 0) or
        not np.isclose(spec.discounts.maximum, 1)):
      raise ValueError('discount_spec must be a BoundedArray in [0, 1].')

    self._spec = spec
    self._episode_length = episode_length
    self._step = 0

  def _generate_fake_observation(self):
    return _generate_from_spec(self._spec.observations)

  def _generate_fake_reward(self):
    return _generate_from_spec(self._spec.rewards)

  def _generate_fake_discount(self):
    return _generate_from_spec(self._spec.discounts)

  def reset(self) -> dm_env.TimeStep:
    observation = self._generate_fake_observation()
    self._step = 1
    return dm_env.restart(observation)

  def step(self, action) -> dm_env.TimeStep:
    # Return a reset timestep if we haven't touched the environment yet.
    if not self._step:
      return self.reset()

    _validate_spec(self._spec.actions, action)

    observation = self._generate_fake_observation()
    reward = self._generate_fake_reward()
    discount = self._generate_fake_discount()

    if self._episode_length and (self._step == self._episode_length):
      self._step = 0
      # We can't use dm_env.termination directly because then the discount
      # wouldn't necessarily conform to the spec (if eg. we want float32).
      return dm_env.TimeStep(dm_env.StepType.LAST, reward, discount,
                             observation)
    else:
      self._step += 1
      return dm_env.transition(
          reward=reward, observation=observation, discount=discount)

  def action_spec(self):
    return self._spec.actions

  def observation_spec(self):
    return self._spec.observations

  def reward_spec(self):
    return self._spec.rewards

  def discount_spec(self):
    return self._spec.discounts


class DiscreteEnvironment(Environment):
  """Discrete state and action fake environment."""

  def __init__(self,
               *,
               num_actions: int = 1,
               num_observations: int = 1,
               action_dtype=np.int32,
               obs_dtype=np.int32,
               reward_dtype=np.float32,
               obs_shape: Sequence[int] = (),
               **kwargs):
    """Initialize the environment."""
    actions = specs.DiscreteArray(num_actions, dtype=action_dtype)
    observations = specs.BoundedArray(
        shape=obs_shape,
        dtype=obs_dtype,
        minimum=obs_dtype(0),
        maximum=obs_dtype(num_observations - 1))
    rewards = specs.Array((), reward_dtype)
    discounts = specs.BoundedArray((), reward_dtype, 0.0, 1.0)

    super().__init__(
        spec=specs.EnvironmentSpec(
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts),
        **kwargs)


class ContinuousEnvironment(Environment):
  """Continuous state and action fake environment."""

  def __init__(self,
               *,
               action_dim: int = 1,
               observation_dim: int = 1,
               bounded: bool = False,
               dtype=np.float32,
               reward_dtype=np.float32,
               **kwargs):
    """Initialize the environment.

    Args:
      action_dim: number of action dimensions.
      observation_dim: number of observation dimensions.
      bounded: whether or not the actions are bounded in [-1, 1].
      dtype: dtype of the action and observation spaces.
      reward_dtype: dtype of the reward and discounts.
      **kwargs: additional kwargs passed to the Environment base class.
    """

    action_shape = () if action_dim == 0 else (action_dim,)
    observation_shape = () if observation_dim == 0 else (observation_dim,)

    observations = specs.Array(observation_shape, dtype)
    rewards = specs.Array((), reward_dtype)
    discounts = specs.BoundedArray((), reward_dtype, 0.0, 1.0)

    if bounded:
      actions = specs.BoundedArray(action_shape, dtype, -1.0, 1.0)
    else:
      actions = specs.Array(action_shape, dtype)

    super().__init__(
        spec=specs.EnvironmentSpec(
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts),
        **kwargs)


def _validate_spec(spec: types.NestedSpec, value: types.NestedArray):
  """Validate a value from a potentially nested spec."""
  tree.assert_same_structure(value, spec)
  tree.map_structure(lambda s, v: s.validate(v), spec, value)


def _generate_from_spec(spec: types.NestedSpec) -> types.NestedArray:
  """Generate a value from a potentially nested spec."""
  return tree.map_structure(lambda s: s.generate_value(), spec)
