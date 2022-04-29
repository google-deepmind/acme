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

"""Unit tests of the LfD adder."""

import collections

from acme import adders
from acme import types
from acme.agents.jax.lfd import lfd_adder
import dm_env
import numpy as np

from absl.testing import absltest


class TestStatisticsAdder(adders.Adder):

  def __init__(self):
    self.counts = collections.defaultdict(int)

  def reset(self):
    pass

  def add_first(self, timestep: dm_env.TimeStep):
    self.counts[int(timestep.observation[0])] += 1

  def add(self,
          action: types.NestedArray,
          next_timestep: dm_env.TimeStep,
          extras: types.NestedArray = ()):
    del action
    del extras
    self.counts[int(next_timestep.observation[0])] += 1


class LfdAdderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._demonstration_episode_type = 1
    self._demonstration_episode_length = 10
    self._collected_episode_type = 2
    self._collected_episode_length = 5

  def generate_episode(self, episode_type, episode_index, length):
    episode = []
    action_dim = 8
    obs_dim = 16
    for k in range(length):
      if k == 0:
        action = None
      else:
        action = np.concatenate([
            np.asarray([episode_type, episode_index], dtype=np.float32),
            np.random.uniform(0., 1., (action_dim - 2,))])
      observation = np.concatenate([
          np.asarray([episode_type, episode_index], dtype=np.float32),
          np.random.uniform(0., 1., (obs_dim - 2,))])
      if k == 0:
        timestep = dm_env.restart(observation)
      elif k == length - 1:
        timestep = dm_env.termination(0., observation)
      else:
        timestep = dm_env.transition(0., observation, 1.)
      episode.append((action, timestep))
    return episode

  def generate_demonstration(self):
    episode_index = 0
    while True:
      episode = self.generate_episode(self._demonstration_episode_type,
                                      episode_index,
                                      self._demonstration_episode_length)
      for x in episode:
        yield x
      episode_index += 1

  def test_adder(self):
    stats_adder = TestStatisticsAdder()
    demonstration_ratio = 0.2
    initial_insert_count = 50
    adder = lfd_adder.LfdAdder(
        stats_adder,
        self.generate_demonstration(),
        initial_insert_count=initial_insert_count,
        demonstration_ratio=demonstration_ratio)

    num_episodes = 100
    for episode_index in range(num_episodes):
      episode = self.generate_episode(self._collected_episode_type,
                                      episode_index,
                                      self._collected_episode_length)
      for k, (action, timestep) in enumerate(episode):
        if k == 0:
          adder.add_first(timestep)
          if episode_index == 0:
            self.assertGreaterEqual(
                stats_adder.counts[self._demonstration_episode_type],
                initial_insert_count - self._demonstration_episode_length)
            self.assertLessEqual(
                stats_adder.counts[self._demonstration_episode_type],
                initial_insert_count + self._demonstration_episode_length)
        else:
          adder.add(action, timestep)

    # Only 2 types of episodes.
    self.assertLen(stats_adder.counts, 2)

    total_count = (stats_adder.counts[self._demonstration_episode_type] +
                   stats_adder.counts[self._collected_episode_type])
    # The demonstration ratio does not account for the initial demonstration
    # insertion. Computes a ratio that takes it into account.
    target_ratio = (
        demonstration_ratio * (float)(total_count - initial_insert_count)
        + initial_insert_count) / (float)(total_count)
    # Effective ratio of demonstrations.
    effective_ratio = (
        float(stats_adder.counts[self._demonstration_episode_type]) /
        float(total_count))
    # Only full episodes can be fed to the adder so the effective ratio
    # might be slightly different from the requested demonstration ratio.
    min_ratio = (target_ratio  -
                 self._demonstration_episode_length / float(total_count))
    max_ratio = (target_ratio +
                 self._demonstration_episode_length / float(total_count))
    self.assertGreaterEqual(effective_ratio, min_ratio)
    self.assertLessEqual(effective_ratio, max_ratio)


if __name__ == '__main__':
  absltest.main()
