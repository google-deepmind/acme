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

"""Tests for tf2_variable_utils."""

from absl.testing import absltest
from acme.testing import fakes
from acme.utils import tf2_utils
from acme.utils import tf2_variable_utils
import numpy as np
import sonnet as snt
import tensorflow as tf


class VariableClientTest(absltest.TestCase):

  def test_update(self):
    # Create two instances of the same model.
    actor_model = snt.nets.MLP([50, 30])
    learner_model = snt.nets.MLP([50, 30])

    # Create variables first.
    input_spec = tf.TensorSpec(shape=(28,), dtype=tf.float32)
    tf2_utils.create_variables(actor_model, [input_spec])
    tf2_utils.create_variables(learner_model, [input_spec])

    # Register them as client and source variables, respectively.
    actor_variables = actor_model.variables
    np_learner_variables = [
        tf2_utils.to_numpy(v) for v in learner_model.variables
    ]
    variable_source = fakes.VariableSource(np_learner_variables)
    variable_client = tf2_variable_utils.VariableClient(
        variable_source, {'policy': actor_variables})

    # Now, given some random batch of test input:
    x = tf.random.normal(shape=(8, 28))

    # Before copying variables, the models have different outputs.
    actor_output = actor_model(x).numpy()
    learner_output = learner_model(x).numpy()
    self.assertFalse(np.allclose(actor_output, learner_output))

    # Update the variable client.
    variable_client.update_and_wait()

    # After copying variables (by updating the client), the models are the same.
    actor_output = actor_model(x).numpy()
    learner_output = learner_model(x).numpy()
    self.assertTrue(np.allclose(actor_output, learner_output))


if __name__ == '__main__':
  absltest.main()
