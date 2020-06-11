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


# Lint as: python3
"""Tests for TF2 savers."""

import os
import time
from unittest import mock

from absl.testing import absltest
from acme import specs
from acme.testing import test_utils
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import paths
import numpy as np
import sonnet as snt
import tensorflow as tf
import tree


class DummySaveable(tf2_savers.TFSaveable):

  _state: tf.Variable

  def __init__(self):
    self._state = tf.Variable(0, dtype=tf.int32)

  @property
  def state(self):
    return {'state': self._state}


class CheckpointerTest(test_utils.TestCase):

  def test_save_and_restore(self):
    """Test that checkpointer correctly calls save and restore."""

    x = tf.Variable(0, dtype=tf.int32)
    directory = self.get_tempdir()
    checkpointer = tf2_savers.Checkpointer(
        objects_to_save={'x': x}, time_delta_minutes=0., directory=directory)

    for _ in range(10):
      saved = checkpointer.save()
      self.assertTrue(saved)
      x.assign_add(1)
      checkpointer.restore()
      np.testing.assert_array_equal(x.numpy(), np.int32(0))

  def test_save_and_new_restore(self):
    """Tests that a fresh checkpointer correctly restores an existing ckpt."""
    with mock.patch.object(paths, 'get_unique_id') as mock_unique_id:
      mock_unique_id.return_value = ('test',)
      x = tf.Variable(0, dtype=tf.int32)
      directory = self.get_tempdir()
      checkpointer1 = tf2_savers.Checkpointer(
          objects_to_save={'x': x}, time_delta_minutes=0., directory=directory)
      checkpointer1.save()
      x.assign_add(1)
      # Simulate a preemption: x is changed, and we make a new Checkpointer.
      checkpointer2 = tf2_savers.Checkpointer(
          objects_to_save={'x': x}, time_delta_minutes=0., directory=directory)
      checkpointer2.restore()
      np.testing.assert_array_equal(x.numpy(), np.int32(0))

  def test_save_and_restore_time_based(self):
    """Test that checkpointer correctly calls save and restore based on time."""

    x = tf.Variable(0, dtype=tf.int32)
    directory = self.get_tempdir()
    checkpointer = tf2_savers.Checkpointer(
        objects_to_save={'x': x}, time_delta_minutes=1., directory=directory)

    with mock.patch.object(time, 'time') as mock_time:
      mock_time.return_value = 0.
      self.assertFalse(checkpointer.save())

      mock_time.return_value = 40.
      self.assertFalse(checkpointer.save())

      mock_time.return_value = 70.
      self.assertTrue(checkpointer.save())
    x.assign_add(1)
    checkpointer.restore()
    np.testing.assert_array_equal(x.numpy(), np.int32(0))

  def test_no_checkpoint(self):
    """Test that checkpointer does nothing when checkpoint=False."""
    num_steps = tf.Variable(0)
    checkpointer = tf2_savers.Checkpointer(
        objects_to_save={'num_steps': num_steps}, enable_checkpointing=False)

    for _ in range(10):
      self.assertFalse(checkpointer.save())
    self.assertIsNone(checkpointer._checkpoint_manager)

  def test_tf_saveable(self):
    x = DummySaveable()

    directory = self.get_tempdir()
    checkpoint_runner = tf2_savers.CheckpointingRunner(
        x, time_delta_minutes=0, directory=directory)
    checkpoint_runner._checkpointer.save()

    x._state.assign_add(1)
    checkpoint_runner._checkpointer.restore()

    np.testing.assert_array_equal(x._state.numpy(), np.int32(0))


class SnapshotterTest(test_utils.TestCase):

  def test_snapshot(self):
    """Test that snapshotter correctly calls saves/restores snapshots."""
    # Create a test network.
    net1 = networks.LayerNormMLP([10, 10])
    spec = specs.Array([10], dtype=np.float32)
    tf2_utils.create_variables(net1, [spec])

    # Save the test network.
    directory = self.get_tempdir()
    objects_to_save = {'net': net1}
    snapshotter = tf2_savers.Snapshotter(objects_to_save, directory=directory)
    snapshotter.save()

    # Reload the test network.
    net2 = tf.saved_model.load(os.path.join(snapshotter.directory, 'net'))
    inputs = tf2_utils.add_batch_dim(tf2_utils.zeros_like(spec))

    with tf.GradientTape() as tape:
      outputs1 = net1(inputs)
      loss1 = tf.math.reduce_sum(outputs1)
      grads1 = tape.gradient(loss1, net1.trainable_variables)

    with tf.GradientTape() as tape:
      outputs2 = net2(inputs)
      loss2 = tf.math.reduce_sum(outputs2)
      grads2 = tape.gradient(loss2, net2.trainable_variables)

    assert np.allclose(outputs1, outputs2)
    assert all(tree.map_structure(np.allclose, list(grads1), list(grads2)))

  def test_snapshot_distribution(self):
    """Test that snapshotter correctly calls saves/restores snapshots."""
    # Create a test network.
    net1 = snt.Sequential([
        networks.LayerNormMLP([10, 10]),
        networks.MultivariateNormalDiagHead(1)
    ])
    spec = specs.Array([10], dtype=np.float32)
    tf2_utils.create_variables(net1, [spec])

    # Save the test network.
    directory = self.get_tempdir()
    objects_to_save = {'net': net1}
    snapshotter = tf2_savers.Snapshotter(objects_to_save, directory=directory)
    snapshotter.save()

    # Reload the test network.
    net2 = tf.saved_model.load(os.path.join(snapshotter.directory, 'net'))
    inputs = tf2_utils.add_batch_dim(tf2_utils.zeros_like(spec))

    with tf.GradientTape() as tape:
      dist1 = net1(inputs)
      loss1 = tf.math.reduce_sum(dist1.mean() + dist1.variance())
      grads1 = tape.gradient(loss1, net1.trainable_variables)

    with tf.GradientTape() as tape:
      dist2 = net2(inputs)
      loss2 = tf.math.reduce_sum(dist2.mean() + dist2.variance())
      grads2 = tape.gradient(loss2, net2.trainable_variables)

    assert all(tree.map_structure(np.allclose, list(grads1), list(grads2)))

  def test_force_snapshot(self):
    """Test that the force feature in Snapshotter.save() works correctly."""
    # Create a test network.
    net = snt.Linear(10)
    spec = specs.Array([10], dtype=np.float32)
    tf2_utils.create_variables(net, [spec])

    # Save the test network.
    directory = self.get_tempdir()
    objects_to_save = {'net': net}
    # Very long time_delta_minutes.
    snapshotter = tf2_savers.Snapshotter(objects_to_save, directory=directory,
                                         time_delta_minutes=1000)
    self.assertTrue(snapshotter.save(force=False))

    # Due to the long time_delta_minutes, only force=True will create a new
    # snapshot. This also checks the default is force=False.
    self.assertFalse(snapshotter.save())
    self.assertTrue(snapshotter.save(force=True))

  def test_rnn_snapshot(self):
    """Test that snapshotter correctly calls saves/restores snapshots on RNNs."""
    # Create a test network.
    net = snt.LSTM(10)
    spec = specs.Array([10], dtype=np.float32)
    tf2_utils.create_variables(net, [spec])

    # Test that if you add some postprocessing without rerunning
    # create_variables, it still works.
    wrapped_net = snt.DeepRNN([net, lambda x: x])

    for net1 in [net, wrapped_net]:
      # Save the test network.
      directory = self.get_tempdir()
      objects_to_save = {'net': net1}
      snapshotter = tf2_savers.Snapshotter(objects_to_save, directory=directory)
      snapshotter.save()

      # Reload the test network.
      net2 = tf.saved_model.load(os.path.join(snapshotter.directory, 'net'))
      inputs = tf2_utils.add_batch_dim(tf2_utils.zeros_like(spec))

      with tf.GradientTape() as tape:
        outputs1, next_state1 = net1(inputs, net1.initial_state(1))
        loss1 = tf.math.reduce_sum(outputs1)
        grads1 = tape.gradient(loss1, net1.trainable_variables)

      with tf.GradientTape() as tape:
        outputs2, next_state2 = net2(inputs, net2.initial_state(1))
        loss2 = tf.math.reduce_sum(outputs2)
        grads2 = tape.gradient(loss2, net2.trainable_variables)

      assert np.allclose(outputs1, outputs2)
      assert np.allclose(tree.flatten(next_state1), tree.flatten(next_state2))
      assert all(tree.map_structure(np.allclose, list(grads1), list(grads2)))


if __name__ == '__main__':
  absltest.main()
