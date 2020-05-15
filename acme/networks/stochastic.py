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

"""Useful sonnet modules to chain after distributional module outputs."""

from acme import types
from acme.utils import tf2_utils
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree

tfd = tfp.distributions


class StochasticMeanHead(snt.Module):
  """Simple sonnet module to produce the mean of a tfp.Distribution."""

  def __call__(self, distribution: tfd.Distribution):
    return distribution.mean()


class StochasticSamplingHead(snt.Module):
  """Simple sonnet module to sample from a tfp.Distribution."""

  def __call__(self, distribution: tfd.Distribution):
    return distribution.sample()


class ExpQWeighedPolicy(snt.Module):
  """Exponentially Q-weighted policy.

  Given a stochastic policy and a critic, returns a (stochastic) policy which
  samples multiple actions from the underlying policy, computes the Q-values for
  each action, and chooses the final action among the sampled ones with
  probability proportional to the exponentiated Q values, tempered by
  a parameter beta.
  """

  def __init__(self,
               actor_network: snt.Module,
               critic_network: snt.Module,
               beta: float = 1.0,
               num_action_samples: int = 16):
    super().__init__(name='ExpQWeighedPolicy')
    self._actor_network = actor_network
    self._critic_network = critic_network
    self._num_action_samples = num_action_samples
    self._beta = beta

  def __call__(self, inputs: types.NestedTensor) -> tf.Tensor:

    batch_size = tree.flatten(inputs)[0].shape[0]
    if batch_size != 1:
      raise NotImplementedError('For now it only supports batch size 1.')

    # Inputs are of size [B, ...]. Here we tile them to be of shape [N, B, ...].
    tiled_inputs = tf2_utils.tile_nested(inputs, self._num_action_samples)
    # Reshape to [N * B, ...].
    merge = lambda x: snt.merge_leading_dims(x, 2)
    tiled_inputs = tree.map_structure(merge, tiled_inputs)

    tiled_actions = self._actor_network(tiled_inputs)

    # Compute Q-values and the resulting tempered probabilities.
    q = self._critic_network(tiled_inputs, tiled_actions)
    boltzmann_probs = tf.nn.softmax(q / self._beta)

    # Resample an action according to the Boltzmann distribution.
    action_idx = tfp.distributions.Categorical(probs=boltzmann_probs).sample()
    action_sample = tiled_actions[action_idx]

    # Add a batch dimension.
    action_sample = action_sample[None]

    return action_sample
