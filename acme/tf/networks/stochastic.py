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


class StochasticModeHead(snt.Module):
  """Simple sonnet module to produce the mode of a tfp.Distribution."""

  def __call__(self, distribution: tfd.Distribution):
    return distribution.mode()


class StochasticMeanHead(snt.Module):
  """Simple sonnet module to produce the mean of a tfp.Distribution."""

  def __call__(self, distribution: tfd.Distribution):
    return distribution.mean()


class StochasticSamplingHead(snt.Module):
  """Simple sonnet module to sample from a tfp.Distribution."""

  def __call__(self, distribution: tfd.Distribution):
    return distribution.sample()


class ExpQWeightedPolicy(snt.Module):
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
    super().__init__(name='ExpQWeightedPolicy')
    self._actor_network = actor_network
    self._critic_network = critic_network
    self._num_action_samples = num_action_samples
    self._beta = beta

  def __call__(self, inputs: types.NestedTensor) -> tf.Tensor:
    # Inputs are of size [B, ...]. Here we tile them to be of shape [N, B, ...].
    tiled_inputs = tf2_utils.tile_nested(inputs, self._num_action_samples)
    shape = tf.shape(tree.flatten(tiled_inputs)[0])
    n, b = shape[0], shape[1]
    tf.debugging.assert_equal(n, self._num_action_samples,
                              'Internal Error. Unexpected tiled_inputs shape.')
    dummy_zeros_n_b = tf.zeros((n, b))
    # Reshape to [N * B, ...].
    merge = lambda x: snt.merge_leading_dims(x, 2)
    tiled_inputs = tree.map_structure(merge, tiled_inputs)

    tiled_actions = self._actor_network(tiled_inputs)

    # Compute Q-values and the resulting tempered probabilities.
    q = self._critic_network(tiled_inputs, tiled_actions)
    boltzmann_probs = tf.nn.softmax(q / self._beta)

    boltzmann_probs = snt.split_leading_dim(boltzmann_probs, dummy_zeros_n_b, 2)
    # [B, N]
    boltzmann_probs = tf.transpose(boltzmann_probs, perm=(1, 0))
    # Resample one action per batch according to the Boltzmann distribution.
    action_idx = tfp.distributions.Categorical(probs=boltzmann_probs).sample()
    # [B, 2], where the first column is 0, 1, 2,... corresponding to indices to
    # the batch dimension.
    action_idx = tf.stack((tf.range(b), action_idx), axis=1)

    tiled_actions = snt.split_leading_dim(tiled_actions, dummy_zeros_n_b, 2)
    action_dim = len(tiled_actions.get_shape().as_list())
    tiled_actions = tf.transpose(tiled_actions,
                                 perm=[1, 0] + list(range(2, action_dim)))
    # [B, ...]
    action_sample = tf.gather_nd(tiled_actions, action_idx)

    return action_sample
