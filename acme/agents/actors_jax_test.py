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

"""Tests for actors_jax."""

from absl.testing import absltest

from acme import environment_loop
from acme import specs
from acme.agents import actors_jax
from acme.testing import fakes
from acme.utils import jax_utils
from acme.utils import jax_variable_utils

import dm_env
import haiku as hk
import jax.numpy as jnp
import numpy as np


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

    def policy(inputs: jnp.ndarray):
      return hk.Sequential([
          hk.Flatten(),
          hk.Linear(env_spec.actions.num_values),
          lambda x: jnp.argmax(x, axis=-1),
      ])(
          inputs)

    policy = hk.transform(policy, apply_rng=True)

    rng = hk.PRNGSequence(1)
    dummy_obs = jax_utils.add_batch_dim(
        jax_utils.zeros_like(env_spec.observations))
    params = policy.init(next(rng), dummy_obs)

    variable_source = fakes.VariableSource(params)
    variable_client = jax_variable_utils.VariableClient(variable_source,
                                                        'policy')

    actor = actors_jax.FeedForwardActor(
        policy.apply, rng=hk.PRNGSequence(1), variable_client=variable_client)

    loop = environment_loop.EnvironmentLoop(environment, actor)
    loop.run(20)


if __name__ == '__main__':
  absltest.main()
