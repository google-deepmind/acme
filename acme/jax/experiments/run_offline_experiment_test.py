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

"""Tests for the run_offline_experiment function."""

from typing import Iterator

from acme import specs
from acme import types
from acme.agents.jax import crr
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.jax.experiments import test_utils as experiment_test_utils
from acme.testing import fakes
from acme.testing import test_utils
import dm_env
from absl.testing import absltest
from absl.testing import parameterized


class RunOfflineExperimentTest(test_utils.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='noeval', num_eval_episodes=0),
      dict(testcase_name='eval', num_eval_episodes=1))
  def test_checkpointing(self, num_eval_episodes: int):
    num_learner_steps = 100

    experiment_config = self._get_experiment_config(
        num_learner_steps=num_learner_steps)

    experiments.run_offline_experiment(
        experiment_config, num_eval_episodes=num_eval_episodes)

    checkpoint_counter = experiment_test_utils.restore_counter(
        experiment_config.checkpointing)
    self.assertIn('learner_steps', checkpoint_counter.get_counts())
    self.assertGreater(checkpoint_counter.get_counts()['learner_steps'], 0)

    # Run the second experiment with the same checkpointing config to verify
    # that it restores from the latest saved checkpoint.
    experiments.run_offline_experiment(
        experiment_config, num_eval_episodes=num_eval_episodes)

    checkpoint_counter = experiment_test_utils.restore_counter(
        experiment_config.checkpointing)
    self.assertIn('learner_steps', checkpoint_counter.get_counts())
    # Verify that the steps done in the first run are taken into account.
    self.assertLessEqual(checkpoint_counter.get_counts()['learner_steps'],
                         num_learner_steps)

  def test_eval_every(self):
    num_learner_steps = 100

    experiment_config = self._get_experiment_config(
        num_learner_steps=num_learner_steps)

    experiments.run_offline_experiment(
        experiment_config, eval_every=70, num_eval_episodes=1)

    checkpoint_counter = experiment_test_utils.restore_counter(
        experiment_config.checkpointing)
    self.assertIn('learner_steps', checkpoint_counter.get_counts())
    self.assertGreater(checkpoint_counter.get_counts()['learner_steps'], 0)
    self.assertLessEqual(checkpoint_counter.get_counts()['learner_steps'],
                         num_learner_steps)

  def _get_experiment_config(
      self, *, num_learner_steps: int) -> experiments.OfflineExperimentConfig:
    def environment_factory(seed: int) -> dm_env.Environment:
      del seed
      return fakes.ContinuousEnvironment(
          episode_length=10, action_dim=3, observation_dim=5)

    environment = environment_factory(seed=1)
    environment_spec = specs.make_environment_spec(environment)

    def demonstration_dataset_factory(
        random_key: jax_types.PRNGKey) -> Iterator[types.Transition]:
      del random_key
      batch_size = 64
      return fakes.transition_iterator_from_spec(environment_spec)(batch_size)

    crr_config = crr.CRRConfig()
    crr_builder = crr.CRRBuilder(
        crr_config, policy_loss_coeff_fn=crr.policy_loss_coeff_advantage_exp)
    checkpointing_config = experiments.CheckpointingConfig(
        directory=self.get_tempdir(), time_delta_minutes=0)
    return experiments.OfflineExperimentConfig(
        builder=crr_builder,
        network_factory=crr.make_networks,
        demonstration_dataset_factory=demonstration_dataset_factory,
        environment_factory=environment_factory,
        max_num_learner_steps=num_learner_steps,
        seed=0,
        environment_spec=environment_spec,
        checkpointing=checkpointing_config,
    )


if __name__ == '__main__':
  absltest.main()
