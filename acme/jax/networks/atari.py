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
from typing import Optional, Tuple, Sequence

from acme.jax.networks import base
from acme.jax.networks import duelling
from acme.jax.networks import embedding
from acme.jax.networks import policy_value
from acme.jax.networks import resnet
from acme.wrappers import observation_action_reward
import haiku as hk
import jax
import jax.numpy as jnp

# Useful type aliases.
Images = jnp.ndarray


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


class DeepAtariTorso(hk.Module):
  """Deep torso for Atari, from the IMPALA paper."""

  def __init__(
      self,
      channels_per_group: Sequence[int] = (16, 32, 32),
      blocks_per_group: Sequence[int] = (2, 2, 2),
      downsampling_strategies: Sequence[resnet.DownsamplingStrategy] = (
          resnet.DownsamplingStrategy.CONV_MAX,) * 3,
      hidden_sizes: Sequence[int] = (256,),
      use_layer_norm: bool = False,
      name: str = 'deep_atari_torso'):
    super().__init__(name=name)
    self._use_layer_norm = use_layer_norm
    self.resnet = resnet.ResNetTorso(
        channels_per_group=channels_per_group,
        blocks_per_group=blocks_per_group,
        downsampling_strategies=downsampling_strategies,
        use_layer_norm=use_layer_norm)
    # Make sure to activate the last layer as this torso is expected to feed
    # into the rest of a bigger network.
    self.mlp_head = hk.nets.MLP(output_sizes=hidden_sizes, activate_final=True)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    output = self.resnet(x)
    output = jax.nn.relu(output)
    output = hk.Flatten(preserve_dims=-3)(output)
    output = self.mlp_head(output)
    return output


class DeepIMPALAAtariNetwork(hk.RNNCore):
  """A recurrent network for use with IMPALA.

  See https://arxiv.org/pdf/1802.01561.pdf for more information.
  """

  def __init__(self, num_actions: int):
    super().__init__(name='impala_atari_network')
    self._embed = embedding.OAREmbedding(
        DeepAtariTorso(use_layer_norm=True), num_actions)
    self._core = hk.GRU(256)
    self._head = policy_value.PolicyValueHead(num_actions)
    self._num_actions = num_actions

  def __call__(self, inputs: observation_action_reward.OAR,
               state: hk.LSTMState) -> base.LSTMOutputs:

    embeddings = self._embed(inputs)  # [B?, D+A+1]
    embeddings, new_state = self._core(embeddings, state)
    logits, value = self._head(embeddings)  # logits: [B?, A], value: [B?, 1]

    return (logits, value), new_state

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
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

  def initial_state(self, batch_size: Optional[int],
                    **unused_kwargs) -> hk.LSTMState:
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
