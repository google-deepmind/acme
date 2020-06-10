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

"""bsuite demonstrations."""

from typing import Any, List

from absl import flags
from bsuite.environments import deep_sea
import dm_env
import numpy as np
import tensorflow as tf
import tree

FLAGS = flags.FLAGS


def _nested_stack(sequence: List[Any]):
  """Stack nested elements in a sequence."""
  return tree.map_structure(lambda *x: np.stack(x), *sequence)


class DemonstrationRecorder:
  """Records demonstrations.

  A demonstration is a (observation, action, reward, discount) tuple where
  every element is a numpy array corresponding to a full episode.
  """

  def __init__(self):
    self._demos = []
    self._reset_episode()

  def step(self, timestep: dm_env.TimeStep, action: np.ndarray):
    reward = np.array(timestep.reward or 0, np.float32)
    self._episode_reward += reward
    self._episode.append((timestep.observation, action, reward,
                          np.array(timestep.discount or 0, np.float32)))

  def record_episode(self):
    self._demos.append(_nested_stack(self._episode))
    self._reset_episode()

  def discard_episode(self):
    self._reset_episode()

  def _reset_episode(self):
    self._episode = []
    self._episode_reward = 0

  @property
  def episode_reward(self):
    return self._episode_reward

  def make_tf_dataset(self):
    types = tree.map_structure(lambda x: x.dtype, self._demos[0])
    shapes = tree.map_structure(lambda x: x.shape, self._demos[0])
    ds = tf.data.Dataset.from_generator(lambda: self._demos, types, shapes)
    return ds.repeat().shuffle(len(self._demos))


def _optimal_deep_sea_policy(environment: deep_sea.DeepSea,
                             timestep: dm_env.TimeStep):
  action = environment._action_mapping[np.where(timestep.observation)]  # pylint: disable=protected-access
  return action[0].astype(np.int32)


def _run_optimal_deep_sea_episode(environment: deep_sea.DeepSea,
                                  recorder: DemonstrationRecorder):
  timestep = environment.reset()
  while timestep.step_type is not dm_env.StepType.LAST:
    action = _optimal_deep_sea_policy(environment, timestep)
    recorder.step(timestep, action)
    timestep = environment.step(action)
  recorder.step(timestep, np.zeros_like(action))


def _make_deep_sea_dataset(environment: deep_sea.DeepSea):
  """Make DeepSea demonstration dataset."""

  recorder = DemonstrationRecorder()

  _run_optimal_deep_sea_episode(environment, recorder)
  assert recorder.episode_reward > 0
  recorder.record_episode()
  return recorder.make_tf_dataset()


def _make_deep_sea_stochastic_dataset(environment: deep_sea.DeepSea):
  """Make stochastic DeepSea demonstration dataset."""

  recorder = DemonstrationRecorder()

  # Use 10*size demos, 80% success, 20% failure.
  num_demos = environment._size * 10  # pylint: disable=protected-access
  num_failures = num_demos // 5
  num_successes = num_demos - num_failures

  successes_saved = 0
  failures_saved = 0
  while (successes_saved < num_successes) or (failures_saved < num_failures):
    _run_optimal_deep_sea_episode(environment, recorder)

    if recorder.episode_reward > 0 and successes_saved < num_successes:
      recorder.record_episode()
      successes_saved += 1
    elif recorder.episode_reward <= 0 and failures_saved < num_failures:
      recorder.record_episode()
      failures_saved += 1
    else:
      recorder.discard_episode()

  return recorder.make_tf_dataset()


def make_dataset(environment: dm_env.Environment):
  """Make bsuite demos for the current task."""

  if FLAGS.bsuite_id.startswith('deep_sea/'):
    assert isinstance(environment, deep_sea.DeepSea)
    return _make_deep_sea_dataset(environment)
  elif FLAGS.bsuite_id.startswith('deep_sea_stochastic/'):
    assert isinstance(environment, deep_sea.DeepSea)
    return _make_deep_sea_stochastic_dataset(environment)
  else:
    raise ValueError('Could not produce demonstrations for {}'
                     .format(FLAGS.bsuite_id))
