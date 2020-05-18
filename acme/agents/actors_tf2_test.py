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

"""Tests for actors_tf2."""

from absl.testing import absltest

from acme import environment_loop
from acme import specs
from acme.agents import actors_tf2
from acme.testing import fakes

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf


def _make_fake_env() -> dm_env.Environment:
  env_spec = specs.EnvironmentSpec(
      observations=specs.Array(shape=(10, 5), dtype=np.float32),
      actions=specs.DiscreteArray(num_values=3),
      rewards=specs.Array(shape=(), dtype=np.float32),
      discounts=specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0., maximum=1.),
  )
  return fakes.Environment(env_spec, episode_length=10)


class ActorTest(absltest.TestCase):

  def test_feedforward(self):
    environment = _make_fake_env()
    env_spec = specs.make_environment_spec(environment)

    network = snt.Sequential([
        snt.Flatten(),
        snt.Linear(env_spec.actions.num_values),
        lambda x: tf.argmax(x, axis=-1, output_type=env_spec.actions.dtype),
    ])

    actor = actors_tf2.FeedForwardActor(network)
    loop = environment_loop.EnvironmentLoop(environment, actor)
    loop.run(20)

  def test_recurrent(self):
    environment = _make_fake_env()
    env_spec = specs.make_environment_spec(environment)

    network = snt.DeepRNN([
        snt.Flatten(),
        snt.Linear(env_spec.actions.num_values),
        lambda x: tf.argmax(x, axis=-1, output_type=env_spec.actions.dtype),
    ])

    actor = actors_tf2.RecurrentActor(network)
    loop = environment_loop.EnvironmentLoop(environment, actor)
    loop.run(20)


if __name__ == '__main__':
  absltest.main()
