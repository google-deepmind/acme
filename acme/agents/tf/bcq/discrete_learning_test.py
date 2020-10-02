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

"""Tests for discrete BCQ learner."""

from absl.testing import absltest

from acme import specs
from acme.agents.tf import bcq
from acme.testing import fakes
from acme.tf import utils as tf2_utils
from acme.tf.networks import discrete as discrete_networks
from acme.utils import counting

import numpy as np
import sonnet as snt


def _make_network(action_spec: specs.DiscreteArray) -> snt.Module:
  return snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, action_spec.num_values]),
  ])


class DiscreteBCQLearnerTest(absltest.TestCase):

  def test_full_learner(self):
    # Create dataset.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)
    dataset = fakes.transition_dataset(environment).batch(2)

    # Build network.
    g_network = _make_network(spec.actions)
    q_network = _make_network(spec.actions)
    network = discrete_networks.DiscreteFilteredQNetwork(g_network=g_network,
                                                         q_network=q_network,
                                                         threshold=0.5)
    tf2_utils.create_variables(network, [spec.observations])

    # Build learner.
    counter = counting.Counter()
    learner = bcq.DiscreteBCQLearner(
        network=network,
        dataset=dataset,
        learning_rate=1e-4,
        discount=0.99,
        importance_sampling_exponent=0.2,
        target_update_period=100,
        counter=counter)

    # Run a learner step.
    learner.step()

    # Check counts from BC and BCQ learners.
    counts = counter.get_counts()
    self.assertEqual(1, counts['bc_steps'])
    self.assertEqual(1, counts['bcq_steps'])

    # Check learner state.
    self.assertEqual(1, learner.state['bc_num_steps'].numpy())
    self.assertEqual(1, learner.state['bcq_num_steps'].numpy())


if __name__ == '__main__':
  absltest.main()
