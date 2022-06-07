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

"""Tests for IQN learner."""

import copy

from acme import specs
from acme.agents.tf import iqn
from acme.testing import fakes
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import counting
import numpy as np
import sonnet as snt

from absl.testing import absltest


def _make_torso_network(num_outputs: int) -> snt.Module:
  """Create torso network (outputs intermediate representation)."""
  return snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([num_outputs])
  ])


def _make_head_network(num_outputs: int) -> snt.Module:
  """Create head network (outputs Q-values)."""
  return snt.nets.MLP([num_outputs])


class IQNLearnerTest(absltest.TestCase):

  def test_full_learner(self):
    # Create dataset.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)
    dataset = fakes.transition_dataset(environment).batch(
        2, drop_remainder=True)

    # Build network.
    network = networks.IQNNetwork(
        torso=_make_torso_network(num_outputs=2),
        head=_make_head_network(num_outputs=spec.actions.num_values),
        latent_dim=2,
        num_quantile_samples=1)
    tf2_utils.create_variables(network, [spec.observations])

    # Build learner.
    counter = counting.Counter()
    learner = iqn.IQNLearner(
        network=network,
        target_network=copy.deepcopy(network),
        dataset=dataset,
        learning_rate=1e-4,
        discount=0.99,
        importance_sampling_exponent=0.2,
        target_update_period=1,
        counter=counter)

    # Run a learner step.
    learner.step()

    # Check counts from IQN learner.
    counts = counter.get_counts()
    self.assertEqual(1, counts['steps'])

    # Check learner state.
    self.assertEqual(1, learner.state['num_steps'].numpy())


if __name__ == '__main__':
  absltest.main()
