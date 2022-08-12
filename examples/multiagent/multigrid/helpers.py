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

"""Helpers for multigrid environment."""

import functools
from typing import Any, Dict, NamedTuple, Sequence

from acme import specs
from acme.agents.jax import ppo
from acme.agents.jax.multiagent.decentralized import factories
from acme.jax import networks as networks_lib
from acme.jax import utils as acme_jax_utils
from acme.multiagent import types as ma_types
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class CategoricalParams(NamedTuple):
  """Parameters for a categorical distribution."""
  logits: jnp.ndarray


def multigrid_obs_preproc(obs: Dict[str, Any],
                          conv_filters: int = 8,
                          conv_kernel: int = 3,
                          scalar_fc: int = 5,
                          scalar_name: str = 'direction',
                          scalar_dim: int = 4) -> jnp.ndarray:
  """Conducts preprocessing on 'multigrid' environment dict observations.

  The preprocessing applied here is similar to those in:
  https://github.com/google-research/google-research/blob/master/social_rl/multiagent_tfagents/multigrid_networks.py

  Args:
    obs: multigrid observation dict, which can include observation inputs such
      as 'image', 'position', and a custom additional observation (defined by
      scalar_name).
    conv_filters: Number of convolution filters.
    conv_kernel: Size of the convolution kernel.
    scalar_fc: Number of neurons in the fully connected layer processing the
      scalar input.
    scalar_name: a special observation key, which is set to
      `direction` in most multigrid environments (and can be overridden here if
      otherwise).
    scalar_dim: Highest possible value for the scalar input. Used to convert to
      one-hot representation.

  Returns:
    out: output observation.
  """

  def _cast_and_scale(x, scale_by=10.0):
    if isinstance(x, jnp.ndarray):
      x = x.astype(jnp.float32)
    return x / scale_by

  outputs = []

  if 'image' in obs.keys():
    image_preproc = hk.Sequential([
        _cast_and_scale,
        hk.Conv2D(output_channels=conv_filters, kernel_shape=conv_kernel),
        jax.nn.relu,
        hk.Flatten()
    ])
    outputs.append(image_preproc(obs['image']))

  if 'position' in obs.keys():
    position_preproc = hk.Sequential([_cast_and_scale, hk.Linear(scalar_fc)])
    outputs.append(position_preproc(obs['position']))

  if scalar_name in obs.keys():
    direction_preproc = hk.Sequential([
        functools.partial(jax.nn.one_hot, num_classes=scalar_dim),
        hk.Flatten(),
        hk.Linear(scalar_fc)
    ])
    outputs.append(direction_preproc(obs[scalar_name]))

  out = jnp.concatenate(outputs, axis=-1)
  return out


def make_multigrid_dqn_networks(
    environment_spec: specs.EnvironmentSpec) -> networks_lib.FeedForwardNetwork:
  """Returns DQN networks used by the agent in the multigrid environment."""
  # Check that multigrid environment is defined with discrete actions, 0-indexed
  assert np.issubdtype(environment_spec.actions.dtype, np.integer), (
      'Expected multigrid environment to have discrete actions with int dtype'
      f' but environment_spec.actions.dtype == {environment_spec.actions.dtype}'
  )
  assert environment_spec.actions.minimum == 0, (
      'Expected multigrid environment to have 0-indexed action indices, but'
      f' environment_spec.actions.minimum == {environment_spec.actions.minimum}'
  )
  num_actions = environment_spec.actions.maximum + 1

  def network(inputs):
    model = hk.Sequential([
        hk.Flatten(),
        hk.nets.MLP([50, 50, num_actions]),
    ])
    processed_inputs = multigrid_obs_preproc(inputs)
    return model(processed_inputs)

  network_hk = hk.without_apply_rng(hk.transform(network))
  dummy_obs = acme_jax_utils.add_batch_dim(
      acme_jax_utils.zeros_like(environment_spec.observations))

  return networks_lib.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, dummy_obs), apply=network_hk.apply)


def make_multigrid_ppo_networks(
    environment_spec: specs.EnvironmentSpec,
    hidden_layer_sizes: Sequence[int] = (64, 64),
) -> ppo.PPONetworks:
  """Returns PPO networks used by the agent in the multigrid environments."""

  # Check that multigrid environment is defined with discrete actions, 0-indexed
  assert np.issubdtype(environment_spec.actions.dtype, np.integer), (
      'Expected multigrid environment to have discrete actions with int dtype'
      f' but environment_spec.actions.dtype == {environment_spec.actions.dtype}'
  )
  assert environment_spec.actions.minimum == 0, (
      'Expected multigrid environment to have 0-indexed action indices, but'
      f' environment_spec.actions.minimum == {environment_spec.actions.minimum}'
  )
  num_actions = environment_spec.actions.maximum + 1

  def forward_fn(inputs):
    processed_inputs = multigrid_obs_preproc(inputs)
    trunk = hk.nets.MLP(hidden_layer_sizes, activation=jnp.tanh)
    h = trunk(processed_inputs)
    logits = hk.Linear(num_actions)(h)
    values = hk.Linear(1)(h)
    values = jnp.squeeze(values, axis=-1)
    return (CategoricalParams(logits=logits), values)

  # Transform into pure functions.
  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

  dummy_obs = acme_jax_utils.zeros_like(environment_spec.observations)
  dummy_obs = acme_jax_utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)
  return make_categorical_ppo_networks(network)  # pylint:disable=undefined-variable


def make_categorical_ppo_networks(
    network: networks_lib.FeedForwardNetwork) -> ppo.PPONetworks:
  """Constructs a PPONetworks for Categorical Policy from FeedForwardNetwork.

  Args:
    network: a transformed Haiku network (or equivalent in other libraries) that
      takes in observations and returns the action distribution and value.

  Returns:
    A PPONetworks instance with pure functions wrapping the input network.
  """

  def log_prob(params: CategoricalParams, action):
    return tfd.Categorical(logits=params.logits).log_prob(action)

  def entropy(params: CategoricalParams):
    return tfd.Categorical(logits=params.logits).entropy()

  def sample(params: CategoricalParams, key: networks_lib.PRNGKey):
    return tfd.Categorical(logits=params.logits).sample(seed=key)

  def sample_eval(params: CategoricalParams, key: networks_lib.PRNGKey):
    del key
    return tfd.Categorical(logits=params.logits).mode()

  return ppo.PPONetworks(
      network=network,
      log_prob=log_prob,
      entropy=entropy,
      sample=sample,
      sample_eval=sample_eval)


def init_default_multigrid_network(
    agent_type: str,
    agent_spec: specs.EnvironmentSpec) -> ma_types.Networks:
  """Returns default networks for multigrid environment."""
  if agent_type == factories.DefaultSupportedAgent.PPO:
    return make_multigrid_ppo_networks(agent_spec)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')
