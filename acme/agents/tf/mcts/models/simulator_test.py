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

"""Tests for simulator.py."""

from absl.testing import absltest
from acme.agents.tf.mcts.models import simulator
from bsuite.environments import catch
import dm_env
import numpy as np


class SimulatorTest(absltest.TestCase):

  def _check_equal(self, a: dm_env.TimeStep, b: dm_env.TimeStep):
    self.assertEqual(a.reward, b.reward)
    self.assertEqual(a.discount, b.discount)
    self.assertEqual(a.step_type, b.step_type)
    np.testing.assert_array_equal(a.observation, b.observation)

  def test_simulator_fidelity(self):
    """Tests whether the simulator match the ground truth."""

    # Given an environment.
    env = catch.Catch()

    # If we instantiate a simulator 'model' of this environment.
    model = simulator.Simulator(env)

    # Then the model and environment should always agree as we step them.
    num_actions = env.action_spec().num_values
    for _ in range(10):
      true_timestep = env.reset()
      self.assertTrue(model.needs_reset)
      model_timestep = model.reset()
      self.assertFalse(model.needs_reset)
      self._check_equal(true_timestep, model_timestep)

      while not true_timestep.last():
        action = np.random.randint(num_actions)
        true_timestep = env.step(action)
        model_timestep = model.step(action)
        self._check_equal(true_timestep, model_timestep)

  def test_checkpointing(self):
    """Tests whether checkpointing restores the state correctly."""
    # Given an environment, and a model based on this environment.
    model = simulator.Simulator(catch.Catch())
    num_actions = model.action_spec().num_values

    model.reset()

    # Now, we save a checkpoint.
    model.save_checkpoint()

    ts = model.step(1)

    # Step the model once and load the checkpoint.
    timestep = model.step(np.random.randint(num_actions))
    model.load_checkpoint()
    self._check_equal(ts, model.step(1))

    while not timestep.last():
      timestep = model.step(np.random.randint(num_actions))

    # The model should require a reset.
    self.assertTrue(model.needs_reset)

    # Once we load checkpoint, the model should no longer require reset.
    model.load_checkpoint()
    self.assertFalse(model.needs_reset)

    # Further steps should agree with the original environment state.
    self._check_equal(ts, model.step(1))


if __name__ == '__main__':
  absltest.main()
