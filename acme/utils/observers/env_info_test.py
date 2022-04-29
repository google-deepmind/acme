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

"""Tests for acme.utils.observers.env_info."""

from acme.utils.observers import env_info
from acme.wrappers import gym_wrapper
import gym
from gym import spaces
import numpy as np

from absl.testing import absltest


class GymEnvWithInfo(gym.Env):

  def __init__(self):
    obs_space = np.ones((10,))
    self.observation_space = spaces.Box(-obs_space, obs_space, dtype=np.float32)
    act_space = np.ones((3,))
    self.action_space = spaces.Box(-act_space, act_space, dtype=np.float32)
    self._step = 0

  def reset(self):
    self._step = 0
    return self.observation_space.sample()

  def step(self, action: np.ndarray):
    self._step += 1
    info = {'survival_bonus': 1}
    if self._step == 1 or self._step == 7:
      info['found_checkpoint'] = 1
    if self._step == 5:
      info['picked_up_an_apple'] = 1
    return self.observation_space.sample(), 0, False, info


class ActionNormTest(absltest.TestCase):

  def test_basic(self):
    env = GymEnvWithInfo()
    env = gym_wrapper.GymWrapper(env)
    observer = env_info.EnvInfoObserver()
    timestep = env.reset()
    observer.observe_first(env, timestep)
    for _ in range(20):
      action = np.zeros((3,))
      timestep = env.step(action)
      observer.observe(env, timestep, action)
    metrics = observer.get_metrics()
    self.assertLen(metrics, 3)
    np.testing.assert_equal(metrics['found_checkpoint'], 2)
    np.testing.assert_equal(metrics['picked_up_an_apple'], 1)
    np.testing.assert_equal(metrics['survival_bonus'], 20)


if __name__ == '__main__':
  absltest.main()
