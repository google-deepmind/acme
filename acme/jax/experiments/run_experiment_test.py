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

"""Tests for the run_experiment function."""

from acme.agents.jax import sac
from acme.jax import experiments
from acme.jax.experiments import test_utils as experiment_test_utils
from acme.testing import fakes
from acme.testing import test_utils
import dm_env
from absl.testing import absltest
from absl.testing import parameterized


class RunExperimentTest(test_utils.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='noeval', num_eval_episodes=0),
      dict(testcase_name='eval', num_eval_episodes=1))
  def test_checkpointing(self, num_eval_episodes: int):
    num_train_steps = 100
    experiment_config = self._get_experiment_config(
        num_train_steps=num_train_steps)

    experiments.run_experiment(
        experiment_config, eval_every=10, num_eval_episodes=num_eval_episodes)

    checkpoint_counter = experiment_test_utils.restore_counter(
        experiment_config.checkpointing)
    self.assertIn('actor_steps', checkpoint_counter.get_counts())
    self.assertGreater(checkpoint_counter.get_counts()['actor_steps'], 0)

    # Run the second experiment with the same checkpointing config to verify
    # that it restores from the latest saved checkpoint.
    experiments.run_experiment(
        experiment_config, eval_every=50, num_eval_episodes=num_eval_episodes)

    checkpoint_counter = experiment_test_utils.restore_counter(
        experiment_config.checkpointing)
    self.assertIn('actor_steps', checkpoint_counter.get_counts())
    # Verify that the steps done in the first run are taken into account.
    self.assertLessEqual(checkpoint_counter.get_counts()['actor_steps'],
                         num_train_steps)

  def test_eval_every(self):
    num_train_steps = 100
    experiment_config = self._get_experiment_config(
        num_train_steps=num_train_steps)

    experiments.run_experiment(
        experiment_config, eval_every=70, num_eval_episodes=1)

    checkpoint_counter = experiment_test_utils.restore_counter(
        experiment_config.checkpointing)
    self.assertIn('actor_steps', checkpoint_counter.get_counts())
    self.assertGreater(checkpoint_counter.get_counts()['actor_steps'], 0)
    self.assertLessEqual(checkpoint_counter.get_counts()['actor_steps'],
                         num_train_steps)

  def _get_experiment_config(
      self, *, num_train_steps: int) -> experiments.ExperimentConfig:
    """Returns a config for a test experiment with the given number of steps."""

    def environment_factory(seed: int) -> dm_env.Environment:
      del seed
      return fakes.ContinuousEnvironment(
          episode_length=10, action_dim=3, observation_dim=5)

    num_train_steps = 100

    sac_config = sac.SACConfig()
    checkpointing_config = experiments.CheckpointingConfig(
        directory=self.get_tempdir(), time_delta_minutes=0)
    return experiments.ExperimentConfig(
        builder=sac.SACBuilder(sac_config),
        environment_factory=environment_factory,
        network_factory=sac.make_networks,
        seed=0,
        max_num_actor_steps=num_train_steps,
        checkpointing=checkpointing_config)


if __name__ == '__main__':
  absltest.main()
