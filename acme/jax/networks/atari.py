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

"""Common networks for Atari.

Glossary of shapes:
- T: Sequence length.
- B: Batch size.
- A: Number of actions.
- D: Embedding size.
- X?: X is optional (e.g. optional batch/sequence dimension).

"""

import functools
from typing import Callable, Sequence, Tuple, Union

from acme.jax.networks import base
from acme.jax.networks import duelling
from acme.jax.networks import embedding
from acme.jax.networks import policy_value
from acme.wrappers import observation_action_reward
import haiku as hk
import jax
import jax.numpy as jnp

# Useful type aliases.
Images = jnp.ndarray
InnerOp = Union[hk.Module, Callable[..., jnp.ndarray]]
MakeInnerOp = Callable[..., InnerOp]
NonLinearity = Callable[[jnp.ndarray], jnp.ndarray]


class AtariTorso(hk.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self):
    super().__init__(name='atari_torso')
    self._network = hk.Sequential([
        hk.Conv2D(32, [8, 8], 4), jax.nn.relu,
        hk.Conv2D(64, [4, 4], 2), jax.nn.relu,
        hk.Conv2D(64, [3, 3], 1), jax.nn.relu
    ])

  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)

    outputs = self._network(inputs)

    if batched_inputs:
      return jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    return jnp.reshape(outputs, [-1])  # [D]


def dqn_atari_network(num_actions: int) -> base.QNetwork:
  """A feed-forward network for use with Ape-X DQN."""

  def network(inputs: Images) -> base.QValues:
    model = hk.Sequential([
        AtariTorso(),
        duelling.DuellingMLP(num_actions, hidden_sizes=[512]),
    ])
    return model(inputs)

  return network


class ResidualBlock(hk.Module):
  """Residual block of operations, e.g. convolutional or MLP."""

  def __init__(self,
               make_inner_op: MakeInnerOp,
               non_linearity: NonLinearity = jax.nn.relu,
               use_layer_norm: bool = False,
               name: str = 'residual_block'):
    super().__init__(name=name)
    self.inner_op1 = make_inner_op()
    self.inner_op2 = make_inner_op()
    self.non_linearity = non_linearity
    self.use_layer_norm = use_layer_norm

    if use_layer_norm:
      self.layernorm1 = hk.LayerNorm(
          axis=(1, 2, 3), create_scale=True, create_offset=True, eps=1e-6)
      self.layernorm2 = hk.LayerNorm(
          axis=(1, 2, 3), create_scale=True, create_offset=True, eps=1e-6)

  def __call__(self, x: jnp.ndarray):
    output = x

    # First layer in residual block.
    if self.use_layer_norm:
      output = self.layernorm1(output)
    output = self.non_linearity(output)
    output = self.inner_op1(output)

    # Second layer in residual block.
    if self.use_layer_norm:
      output = self.layernorm2(output)
    output = self.non_linearity(output)
    output = self.inner_op2(output)
    return x + output


class ResNetTorso(hk.Module):
  """ResNetTorso for visual inputs, inspired by the IMPALA paper."""

  def __init__(self,
               channels_per_group: Sequence[int] = (16, 32, 32),
               blocks_per_group: Sequence[int] = (2, 2, 2),
               use_layer_norm: bool = False,
               name: str = 'resnet_torso'):
    super().__init__(name=name)
    self._channels_per_group = channels_per_group
    self._blocks_per_group = blocks_per_group
    self._use_layer_norm = use_layer_norm

    if len(channels_per_group) != len(blocks_per_group):
      raise ValueError(
          'Length of channels_per_group and blocks_per_group must be equal. '
          f'Got channels_per_group={channels_per_group} and '
          f'blocks_per_group={blocks_per_group}.')

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    output = inputs
    channels_and_blocks = zip(self._channels_per_group, self._blocks_per_group)

    for i, (num_channels, num_blocks) in enumerate(channels_and_blocks):
      output = hk.Conv2D(
          num_channels, kernel_shape=3, stride=1, padding='SAME')(
              output)
      output = hk.max_pool(
          output,
          window_shape=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME')

      for j in range(num_blocks):
        output = ResidualBlock(
            make_inner_op=functools.partial(
                hk.Conv2D, output_channels=num_channels, kernel_shape=3),
            use_layer_norm=self._use_layer_norm,
            name=f'residual_{i}_{j}',
        )(
            output)

    return output


class DeepAtariTorso(hk.Module):
  """Deep torso for Atari, from the IMPALA paper."""

  def __init__(self, name: str = 'deep_atari_torso'):
    super().__init__(name=name)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    output = ResNetTorso(
        channels_per_group=(16, 32, 32), blocks_per_group=(2, 2, 2))(
            x)
    output = jax.nn.relu(output)
    output = hk.Flatten()(output)
    output = hk.Linear(256)(output)
    output = jax.nn.relu(output)
    return output


class DeepIMPALAAtariNetwork(hk.RNNCore):
  """A recurrent network for use with IMPALA.

  See https://arxiv.org/pdf/1802.01561.pdf for more information.
  """

  def __init__(self, num_actions: int):
    super().__init__(name='impala_atari_network')
    self._embed = embedding.OAREmbedding(DeepAtariTorso(), num_actions)
    self._core = hk.LSTM(256)
    self._head = policy_value.PolicyValueHead(num_actions)
    self._num_actions = num_actions

  def __call__(self, inputs: observation_action_reward.OAR,
               state: hk.LSTMState) -> base.LSTMOutputs:

    embeddings = self._embed(inputs)  # [B?, D+A+1]
    embeddings, new_state = self._core(embeddings, state)
    logits, value = self._head(embeddings)  # logits: [B?, A], value: [B?, 1]

    return (logits, value), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(self, inputs: observation_action_reward.OAR,
             state: hk.LSTMState) -> base.LSTMOutputs:
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    embeddings = self._embed(inputs)
    embeddings, new_states = hk.static_unroll(self._core, embeddings, state)
    logits, values = self._head(embeddings)

    return (logits, values), new_states


class R2D2AtariNetwork(hk.RNNCore):
  """A duelling recurrent network for use with Atari observations as seen in R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self, num_actions: int):
    super().__init__(name='r2d2_atari_network')
    self._embed = embedding.OAREmbedding(DeepAtariTorso(), num_actions)
    self._core = hk.LSTM(512)
    self._duelling_head = duelling.DuellingMLP(num_actions, hidden_sizes=[512])
    self._num_actions = num_actions

  def __call__(
      self,
      inputs: observation_action_reward.OAR,  # [B, ...]
      state: hk.LSTMState  # [B, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    embeddings = self._embed(inputs)  # [B, D+A+1]
    core_outputs, new_state = self._core(embeddings, state)
    q_values = self._duelling_head(core_outputs)
    return q_values, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,  # [T, B, ...]
      state: hk.LSTMState  # [T, ...]
  ) -> Tuple[base.QValues, hk.LSTMState]:
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    embeddings = hk.BatchApply(self._embed)(inputs)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
    q_values = hk.BatchApply(self._duelling_head)(core_outputs)  # [T, B, A]
    return q_values, new_states
