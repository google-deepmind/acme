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

"""Tests for actors."""
from typing import Optional, Tuple

from acme import environment_loop
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.jax import utils
from acme.jax import variable_utils
from acme.testing import fakes
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


def _make_fake_env() -> dm_env.Environment:
  env_spec = specs.EnvironmentSpec(
      observations=specs.Array(shape=(10, 5), dtype=np.float32),
      actions=specs.DiscreteArray(num_values=3),
      rewards=specs.Array(shape=(), dtype=np.float32),
      discounts=specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0., maximum=1.),
  )
  return fakes.Environment(env_spec, episode_length=10)


class ActorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('policy', False),
      ('policy_with_extras', True))
  def test_feedforward(self, has_extras):
    environment = _make_fake_env()
    env_spec = specs.make_environment_spec(environment)

    def policy(inputs: jnp.ndarray):
      action_values = hk.Sequential([
          hk.Flatten(),
          hk.Linear(env_spec.actions.num_values),
      ])(
          inputs)
      action = jnp.argmax(action_values, axis=-1)
      if has_extras:
        return action, (action_values,)
      else:
        return action

    policy = hk.transform(policy)

    rng = hk.PRNGSequence(1)
    dummy_obs = utils.add_batch_dim(utils.zeros_like(env_spec.observations))
    params = policy.init(next(rng), dummy_obs)

    variable_source = fakes.VariableSource(params)
    variable_client = variable_utils.VariableClient(variable_source, 'policy')

    if has_extras:
      actor_core = actor_core_lib.batched_feed_forward_with_extras_to_actor_core(
          policy.apply)
    else:
      actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
          policy.apply)
    actor = actors.GenericActor(
        actor_core,
        random_key=jax.random.PRNGKey(1),
        variable_client=variable_client)

    loop = environment_loop.EnvironmentLoop(environment, actor)
    loop.run(20)


def _transform_without_rng(f):
  return hk.without_apply_rng(hk.transform(f))


class RecurrentActorTest(absltest.TestCase):

  def test_recurrent(self):
    environment = _make_fake_env()
    env_spec = specs.make_environment_spec(environment)
    output_size = env_spec.actions.num_values
    obs = utils.add_batch_dim(utils.zeros_like(env_spec.observations))
    rng = hk.PRNGSequence(1)

    @_transform_without_rng
    def network(inputs: jnp.ndarray, state: hk.LSTMState):
      return hk.DeepRNN([hk.Reshape([-1], preserve_dims=1),
                         hk.LSTM(output_size)])(inputs, state)

    @_transform_without_rng
    def initial_state(batch_size: Optional[int] = None):
      network = hk.DeepRNN([hk.Reshape([-1], preserve_dims=1),
                            hk.LSTM(output_size)])
      return network.initial_state(batch_size)

    initial_state = initial_state.apply(initial_state.init(next(rng)), 1)
    params = network.init(next(rng), obs, initial_state)

    def policy(
        params: jnp.ndarray,
        key: jnp.ndarray,
        observation: jnp.ndarray,
        core_state: hk.LSTMState
    ) -> Tuple[jnp.ndarray, hk.LSTMState]:
      del key  # Unused for test-case deterministic policy.
      action_values, core_state = network.apply(params, observation, core_state)
      actions = jnp.argmax(action_values, axis=-1)
      return actions, core_state

    variable_source = fakes.VariableSource(params)
    variable_client = variable_utils.VariableClient(variable_source, 'policy')

    actor_core = actor_core_lib.batched_recurrent_to_actor_core(
        policy, initial_state)
    actor = actors.GenericActor(actor_core, jax.random.PRNGKey(1),
                                variable_client)

    loop = environment_loop.EnvironmentLoop(environment, actor)
    loop.run(20)


if __name__ == '__main__':
  absltest.main()
