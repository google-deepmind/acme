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

import threading
from typing import List, Mapping, Optional, Sequence, Callable, Iterator

from acme import core
from acme import specs
from acme import types
from acme import wrappers
import dm_env
import numpy as np
import reverb
from rlds import rlds_types
import tensorflow as tf
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

  def update(self, wait: bool = False):
    self.num_updates += 1


class VariableSource(core.VariableSource):
  """Fake variable source."""

  def __init__(self,
               variables: Optional[types.NestedArray] = None,
               barrier: Optional[threading.Barrier] = None,
               use_default_key: bool = True):
    # Add dummy variables so we can expose them in get_variables.
    if use_default_key:
      self._variables = {'policy': [] if variables is None else variables}
    else:
      self._variables = variables
    self._barrier = barrier

  def get_variables(self, names: List[str]) -> List[types.NestedArray]:
    if self._barrier is not None:
      self._barrier.wait()
    return [self._variables[name] for name in names]


class Learner(core.Learner, VariableSource):
  """Fake Learner."""

  def __init__(self,
               variables: Optional[types.NestedArray] = None,
               barrier: Optional[threading.Barrier] = None):
    super().__init__(variables=variables, barrier=barrier)
    self.step_counter = 0

  def step(self):
    self.step_counter += 1


class Environment(dm_env.Environment):
  """A fake environment with a given spec."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      *,
      episode_length: int = 25,
  ):
    # Assert that the discount spec is a BoundedArray with range [0, 1].
    def check_discount_spec(path, discount_spec):
      if (not isinstance(discount_spec, specs.BoundedArray) or
          not np.isclose(discount_spec.minimum, 0) or
          not np.isclose(discount_spec.maximum, 1)):
        if path:
          path_str = ' ' + '/'.join(str(p) for p in path)
        else:
          path_str = ''
        raise ValueError(
            'discount_spec {}isn\'t a BoundedArray in [0, 1].'.format(path_str))

    tree.map_structure_with_path(check_discount_spec, spec.discounts)

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


class _BaseDiscreteEnvironment(Environment):
  """Discrete action fake environment."""

  def __init__(self,
               *,
               num_actions: int = 1,
               action_dtype=np.int32,
               observation_spec: types.NestedSpec,
               discount_spec: Optional[types.NestedSpec] = None,
               reward_spec: Optional[types.NestedSpec] = None,
               **kwargs):
    """Initialize the environment."""
    if reward_spec is None:
      reward_spec = specs.Array((), np.float32)

    if discount_spec is None:
      discount_spec = specs.BoundedArray((), np.float32, 0.0, 1.0)

    actions = specs.DiscreteArray(num_actions, dtype=action_dtype)

    super().__init__(
        spec=specs.EnvironmentSpec(
            observations=observation_spec,
            actions=actions,
            rewards=reward_spec,
            discounts=discount_spec),
        **kwargs)


class DiscreteEnvironment(_BaseDiscreteEnvironment):
  """Discrete state and action fake environment."""

  def __init__(self,
               *,
               num_actions: int = 1,
               num_observations: int = 1,
               action_dtype=np.int32,
               obs_dtype=np.int32,
               obs_shape: Sequence[int] = (),
               discount_spec: Optional[types.NestedSpec] = None,
               reward_spec: Optional[types.NestedSpec] = None,
               **kwargs):
    """Initialize the environment."""
    observations_spec = specs.BoundedArray(
        shape=obs_shape,
        dtype=obs_dtype,
        minimum=obs_dtype(0),
        maximum=obs_dtype(num_observations - 1))

    super().__init__(
        num_actions=num_actions,
        action_dtype=action_dtype,
        observation_spec=observations_spec,
        discount_spec=discount_spec,
        reward_spec=reward_spec,
        **kwargs)


class NestedDiscreteEnvironment(_BaseDiscreteEnvironment):
  """Discrete action fake environment with nested discrete state."""

  def __init__(self,
               *,
               num_observations: Mapping[str, int],
               num_actions: int = 1,
               action_dtype=np.int32,
               obs_dtype=np.int32,
               obs_shape: Sequence[int] = (),
               discount_spec: Optional[types.NestedSpec] = None,
               reward_spec: Optional[types.NestedSpec] = None,
               **kwargs):
    """Initialize the environment."""

    observations_spec = {}
    for key in num_observations:
      observations_spec[key] = specs.BoundedArray(
          shape=obs_shape,
          dtype=obs_dtype,
          minimum=obs_dtype(0),
          maximum=obs_dtype(num_observations[key] - 1))

    super().__init__(
        num_actions=num_actions,
        action_dtype=action_dtype,
        observation_spec=observations_spec,
        discount_spec=discount_spec,
        reward_spec=reward_spec,
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


def _normalize_array(array: specs.Array) -> specs.Array:
  """Converts bounded arrays with (-inf,+inf) bounds to unbounded arrays.

  The returned array should be mostly equivalent to the input, except that
  `generate_value()` returns -infs on arrays bounded to (-inf,+inf) and zeros
  on unbounded arrays.

  Args:
    array: the array to be normalized.

  Returns:
    normalized array.
  """
  if isinstance(array, specs.DiscreteArray):
    return array
  if not isinstance(array, specs.BoundedArray):
    return array
  if not (array.minimum == float('-inf')).all():
    return array
  if not (array.maximum == float('+inf')).all():
    return array
  return specs.Array(array.shape, array.dtype, array.name)


def _generate_from_spec(spec: types.NestedSpec) -> types.NestedArray:
  """Generate a value from a potentially nested spec."""
  return tree.map_structure(lambda s: _normalize_array(s).generate_value(),
                            spec)


def transition_dataset_from_spec(
    spec: specs.EnvironmentSpec) -> tf.data.Dataset:
  """Constructs fake dataset of Reverb N-step transition samples.

  Args:
    spec: Constructed fake transitions match the provided specification.

  Returns:
    tf.data.Dataset that produces the same fake N-step transition ReverbSample
    object indefinitely.
  """

  observation = _generate_from_spec(spec.observations)
  action = _generate_from_spec(spec.actions)
  reward = _generate_from_spec(spec.rewards)
  discount = _generate_from_spec(spec.discounts)
  data = types.Transition(observation, action, reward, discount, observation)

  info = tree.map_structure(
      lambda tf_dtype: tf.ones([], tf_dtype.as_numpy_dtype),
      reverb.SampleInfo.tf_dtypes())
  sample = reverb.ReplaySample(info=info, data=data)

  return tf.data.Dataset.from_tensors(sample).repeat()


def transition_dataset(environment: dm_env.Environment) -> tf.data.Dataset:
  """Constructs fake dataset of Reverb N-step transition samples.

  Args:
    environment: Constructed fake transitions will match the specification of
      this environment.

  Returns:
    tf.data.Dataset that produces the same fake N-step transition ReverbSample
    object indefinitely.
  """
  return transition_dataset_from_spec(specs.make_environment_spec(environment))


def transition_iterator_from_spec(
    spec: specs.EnvironmentSpec) -> Callable[[int], Iterator[types.Transition]]:
  """Constructs fake iterator of transitions.

  Args:
    spec: Constructed fake transitions match the provided specification..

  Returns:
    A callable that given a batch_size returns an iterator of transitions.
  """

  observation = _generate_from_spec(spec.observations)
  action = _generate_from_spec(spec.actions)
  reward = _generate_from_spec(spec.rewards)
  discount = _generate_from_spec(spec.discounts)
  data = types.Transition(observation, action, reward, discount, observation)

  dataset = tf.data.Dataset.from_tensors(data).repeat()

  return lambda batch_size: dataset.batch(batch_size).as_numpy_iterator()


def transition_iterator(
    environment: dm_env.Environment
) -> Callable[[int], Iterator[types.Transition]]:
  """Constructs fake iterator of transitions.

  Args:
    environment: Constructed fake transitions will match the specification of
      this environment.

  Returns:
    A callable that given a batch_size returns an iterator of transitions.
  """
  return transition_iterator_from_spec(specs.make_environment_spec(environment))


def fake_atari_wrapped(episode_length: int = 10,
                       oar_wrapper: bool = False) -> dm_env.Environment:
  """Builds fake version of the environment to be used by tests.

  Args:
    episode_length: The length of episodes produced by this environment.
    oar_wrapper: Should ObservationActionRewardWrapper be applied.

  Returns:
    Fake version of the environment equivalent to the one returned by
    env_loader.load_atari_wrapped
  """
  env = DiscreteEnvironment(
      num_actions=18,
      num_observations=2,
      obs_shape=(84, 84, 4),
      obs_dtype=np.float32,
      episode_length=episode_length)

  if oar_wrapper:
    env = wrappers.ObservationActionRewardWrapper(env)
  return env


def rlds_dataset_from_env_spec(
    spec: specs.EnvironmentSpec,
    *,
    episode_count: int = 10,
    episode_length: int = 25,
) -> tf.data.Dataset:
  """Constructs a fake RLDS dataset with the given spec.

  Args:
    spec: specification to use for generation of fake steps.
    episode_count: number of episodes in the dataset.
    episode_length: length of the episode in the dataset.

  Returns:
    a fake RLDS dataset.
  """

  fake_steps = {
      rlds_types.OBSERVATION:
          ([_generate_from_spec(spec.observations)] * episode_length),
      rlds_types.ACTION: ([_generate_from_spec(spec.actions)] * episode_length),
      rlds_types.REWARD: ([_generate_from_spec(spec.rewards)] * episode_length),
      rlds_types.DISCOUNT:
          ([_generate_from_spec(spec.discounts)] * episode_length),
      rlds_types.IS_TERMINAL: [False] * (episode_length - 1) + [True],
      rlds_types.IS_FIRST: [True] + [False] * (episode_length - 1),
      rlds_types.IS_LAST: [False] * (episode_length - 1) + [True],
  }
  steps_dataset = tf.data.Dataset.from_tensor_slices(fake_steps)

  return tf.data.Dataset.from_tensor_slices(
      {rlds_types.STEPS: [steps_dataset] * episode_count})
