# Lint as: python3
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

"""Test networks for building recurrent agents."""

import os

from absl.testing import absltest
from acme import specs
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.tf.networks import recurrence

import numpy as np
import sonnet as snt
import tensorflow as tf
import tree


# Simple critic-like modules for testing.
class Critic(snt.Module):

  def __call__(self, o, a):
    return o * a


class RNNCritic(snt.RNNCore):

  def __call__(self, o, a, prev_state):
    return o * a, prev_state

  def initial_state(self, batch_size):
    return ()


class NetsTest(tf.test.TestCase):

  def test_criticdeeprnn_snapshot(self):
    """Test that CriticDeepRNN works correctly with snapshotting."""
    # Create a test network.
    critic = Critic()
    rnn_critic = RNNCritic()

    for base_net in [critic, rnn_critic]:
      net = recurrence.CriticDeepRNN([base_net, snt.LSTM(10)])
      obs = specs.Array([10], dtype=np.float32)
      actions = specs.Array([10], dtype=np.float32)
      spec = [obs, actions]
      tf2_utils.create_variables(net, spec)

      # Test that if you add some postprocessing without rerunning
      # create_variables, it still works.
      wrapped_net = recurrence.CriticDeepRNN([net, lambda x: x])

      for curr_net in [net, wrapped_net]:
        # Save the test network.
        directory = absltest.get_default_test_tmpdir()
        objects_to_save = {'net': curr_net}
        snapshotter = tf2_savers.Snapshotter(
            objects_to_save, directory=directory)
        snapshotter.save()

        # Reload the test network.
        net2 = tf.saved_model.load(os.path.join(snapshotter.directory, 'net'))

        obs = tf.ones((2, 10))
        actions = tf.ones((2, 10))
        state = curr_net.initial_state(2)
        outputs1, next_state1 = curr_net(obs, actions, state)
        outputs2, next_state2 = net2(obs, actions, state)

        assert np.allclose(outputs1, outputs2)
        assert np.allclose(tree.flatten(next_state1), tree.flatten(next_state2))


if __name__ == '__main__':
  absltest.main()
