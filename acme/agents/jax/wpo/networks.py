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

"""WPO network definitions."""

import dataclasses
from typing import Callable, NamedTuple, Sequence

from acme import specs
from acme.agents.jax.wpo import types
from acme.jax import networks as networks_lib
from acme.jax import utils
import chex
import haiku as hk
import haiku.initializers as hk_init
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
DistributionOrArray = tfd.Distribution | jnp.ndarray


class WPONetworkParams(NamedTuple):
  policy_head: hk.Params | None = None
  critic_head: hk.Params | None = None
  torso: hk.Params | None = None
  torso_initial_state: hk.Params | None = None


@dataclasses.dataclass
class UnrollableNetwork:
  """Network that can unroll over an input sequence."""
  init: Callable[[networks_lib.PRNGKey, types.Observation, hk.LSTMState],
                 hk.Params]
  apply: Callable[[hk.Params, types.Observation, hk.LSTMState],
                  tuple[jnp.ndarray, hk.LSTMState]]
  unroll: Callable[[hk.Params, types.Observation, hk.LSTMState],
                   tuple[jnp.ndarray, hk.LSTMState]]
  initial_state_fn_init: Callable[[networks_lib.PRNGKey, int | None],
                                  hk.Params]
  initial_state_fn: Callable[[hk.Params, int | None], hk.LSTMState]


@dataclasses.dataclass
class WPONetworks:
  """Network for the WPO agent."""
  policy_head: hk.Transformed | None = None
  critic_head: hk.Transformed | None = None
  torso: UnrollableNetwork | None = None
  dynamics_model: UnrollableNetwork | None = None

  def policy_head_apply(self, params: WPONetworkParams,
                        obs_embedding: types.ObservationEmbedding):
    if self.policy_head is None:
      raise ValueError('Policy head is not initialized.')
    return self.policy_head.apply(params.policy_head, obs_embedding)

  def critic_head_apply(self, params: WPONetworkParams,
                        obs_embedding: types.ObservationEmbedding,
                        actions: types.Action):
    if self.critic_head is None:
      raise ValueError('Critic head is not initialized.')
    return self.critic_head.apply(params.critic_head, obs_embedding, actions)

  def torso_unroll(self, params: WPONetworkParams,
                   observations: types.Observation, state: hk.LSTMState):
    if self.torso is None:
      raise ValueError('Torso is not initialized.')
    return self.torso.unroll(params.torso, observations, state)


def init_params(
    networks: WPONetworks,
    spec: specs.EnvironmentSpec,
    random_key: types.RNGKey,
    add_batch_dim: bool = False,
) -> tuple[WPONetworkParams, hk.LSTMState]:
  """Initialize the parameters of a WPO network."""

  rng_keys = jax.random.split(random_key, 6)

  # Create a dummy observation/action to initialize network parameters.
  observations, actions = utils.zeros_like((spec.observations, spec.actions))

  # Add batch dimensions if necessary by the scope that is calling this init.
  if add_batch_dim:
    observations, actions = utils.add_batch_dim((observations, actions))

  # Initialize the state torso parameters and create a dummy core state.
  batch_size = 1 if add_batch_dim else None
  if networks.torso is None:
    raise ValueError('Torso is not initialized.')
  params_torso_initial_state = networks.torso.initial_state_fn_init(
      rng_keys[0], batch_size)
  state = networks.torso.initial_state_fn(
      params_torso_initial_state, batch_size)

  # Initialize the core and unroll one step to create a dummy core output.
  # The input to the core is the current action and the next observation.
  params_torso = networks.torso.init(rng_keys[1], observations, state)
  embeddings, _ = networks.torso.apply(params_torso, observations, state)

  # Initialize the policy and critic heads by passing in the dummy embedding.
  params_policy_head, params_critic_head = {}, {}  # Cannot be None for BIT.
  if networks.policy_head:
    params_policy_head = networks.policy_head.init(rng_keys[2], embeddings)
  if networks.critic_head:
    params_critic_head = networks.critic_head.init(rng_keys[3], embeddings,
                                                   actions)

  params = WPONetworkParams(
      policy_head=params_policy_head,
      critic_head=params_critic_head,
      torso=params_torso,
      torso_initial_state=params_torso_initial_state)

  return params, state


def make_unrollable_network(
    make_core_module: Callable[[], hk.RNNCore] = hk.IdentityCore,
    make_feedforward_module: Callable[[], hk.SupportsCall] | None = None,
    make_initial_state_fn: Callable[[], hk.SupportsCall] | None = None,
) -> UnrollableNetwork:
  """Produces an UnrollableNetwork and a state initializing hk.Transformed."""

  def default_initial_state_fn(batch_size: int | None = None) -> jnp.ndarray:
    return make_core_module().initial_state(batch_size)

  def _apply_core_fn(observation: types.Observation,
                     state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if make_feedforward_module:
      observation = make_feedforward_module()(observation)
    return make_core_module()(observation, state)

  def _unroll_core_fn(observation: types.Observation,
                      state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if make_feedforward_module:
      observation = make_feedforward_module()(observation)
    return hk.dynamic_unroll(make_core_module(), observation, state)

  if make_initial_state_fn:
    initial_state_fn = make_initial_state_fn()
  else:
    initial_state_fn = default_initial_state_fn

  # Transform module functions into pure functions.
  hk_initial_state_fn = hk.without_apply_rng(hk.transform(initial_state_fn))
  apply_core = hk.without_apply_rng(hk.transform(_apply_core_fn))
  unroll_core = hk.without_apply_rng(hk.transform(_unroll_core_fn))

  # Pack all core network pure functions into a single convenient container.
  return UnrollableNetwork(
      init=apply_core.init,
      apply=apply_core.apply,
      unroll=unroll_core.apply,
      initial_state_fn_init=hk_initial_state_fn.init,
      initial_state_fn=hk_initial_state_fn.apply)


def make_control_networks(
    environment_spec: specs.EnvironmentSpec,
    *,
    with_recurrence: bool = False,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    policy_init_scale: float = 0.7,
) -> WPONetworks:
  """Creates WPONetworks to be used DM Control suite tasks."""

  # Unpack the environment spec to get appropriate shapes, dtypes, etc.
  num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

  # Factory to create the core hk.Module. Must be a factory as the module must
  # be initialized within a hk.transform scope.
  if with_recurrence:
    make_core_module = lambda: GRUWithSkip(16)
  else:
    make_core_module = hk.IdentityCore

  def policy_fn(observation: types.NestedArray) -> tfd.Distribution:
    embedding = networks_lib.LayerNormMLP(
        policy_layer_sizes, activate_final=True)(
            observation)
    return networks_lib.MultivariateNormalDiagHead(
        num_dimensions, init_scale=policy_init_scale)(
            embedding)

  def critic_fn(observation: types.NestedArray,
                action: types.NestedArray) -> DistributionOrArray:
    # Action is clipped to avoid critic extrapolations outside the spec range.
    clipped_action = networks_lib.ClipToSpec(environment_spec.actions)(action)
    inputs = jnp.concatenate([observation, clipped_action], axis=-1)
    embedding = networks_lib.LayerNormMLP(
        critic_layer_sizes, activate_final=True)(
            inputs)

    return hk.Linear(
        output_size=1, w_init=hk_init.TruncatedNormal(0.01))(
            embedding)

  # Create unrollable torso.
  torso = make_unrollable_network(make_core_module=make_core_module)

  # Create WPONetworks to add functionality required by the agent.
  return WPONetworks(
      policy_head=hk.without_apply_rng(hk.transform(policy_fn)),
      critic_head=hk.without_apply_rng(hk.transform(critic_fn)),
      torso=torso)


def add_batch(nest, batch_size: int | None):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree.map(broadcast, nest)


def w_init_identity(shape: Sequence[int], dtype) -> jnp.ndarray:
  chex.assert_equal(len(shape), 2)
  chex.assert_equal(shape[0], shape[1])
  return jnp.eye(shape[0], dtype=dtype)


class IdentityRNN(hk.RNNCore):
  r"""Basic fully-connected RNN core with identity initialization.

  Given :math:`x_t` and the previous hidden state :math:`h_{t-1}` the
  core computes
  .. math::
     h_t = \operatorname{ReLU}(w_i x_t + b_i + w_h h_{t-1} + b_h)
  The output is equal to the new state, :math:`h_t`.

  Initialized using the strategy described in:
    https://arxiv.org/pdf/1504.00941.pdf
  """

  def __init__(self,
               hidden_size: int,
               hidden_scale: float = 1e-2,
               name: str | None = None):
    """Constructs a vanilla RNN core.

    Args:
      hidden_size: Hidden layer size.
      hidden_scale: Scalar multiplying the hidden-to-hidden matmul.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._initial_state = jnp.zeros([hidden_size])
    self._hidden_scale = hidden_scale
    self._input_to_hidden = hk.Linear(hidden_size)
    self._hidden_to_hidden = hk.Linear(
        hidden_size, with_bias=True, w_init=w_init_identity)

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    out = jax.nn.relu(
        self._input_to_hidden(inputs) +
        self._hidden_scale * self._hidden_to_hidden(prev_state))
    return out, out

  def initial_state(self, batch_size: int | None):
    state = self._initial_state
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class GRU(hk.GRU):
  """GRU with an identity initialization."""

  def __init__(self, hidden_size: int, name: str | None = None):

    def b_init(unused_size: Sequence[int], dtype) -> jnp.ndarray:
      """Initializes the biases so the GRU ignores the state and acts as a tanh."""
      return jnp.concatenate([
          +2 * jnp.ones([hidden_size], dtype=dtype),
          -2 * jnp.ones([hidden_size], dtype=dtype),
          jnp.zeros([hidden_size], dtype=dtype)
      ])

    super().__init__(hidden_size=hidden_size, b_init=b_init, name=name)


class GRUWithSkip(hk.GRU):
  """GRU with a skip-connection from input to output."""

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    outputs, state = super().__call__(inputs, prev_state)
    outputs = jnp.concatenate([inputs, outputs], axis=-1)
    return outputs, state


class Conv2DLSTMWithSkip(hk.Conv2DLSTM):
  """Conv2DLSTM with a skip-connection from input to output."""

  def __call__(self, inputs: jnp.ndarray, state: jnp.ndarray):
    outputs, state = super().__call__(inputs, state)  # pytype: disable=wrong-arg-types  # jax-ndarray
    outputs = jnp.concatenate([inputs, outputs], axis=-1)
    return outputs, state
