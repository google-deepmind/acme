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

"""Utility functions for SVG0 algorithm with priors."""

import collections
from typing import Tuple, Optional, Dict, Iterable

from acme import types
from acme.tf import utils as tf2_utils

import sonnet as snt
import tensorflow as tf
import tree


class OnlineTargetPiQ(snt.Module):
  """Core to unroll online and target policies and Q functions at once.

  A core that runs online and target policies and Q functions. This can be more
  efficient if the core needs to be unrolled across time and called many times.
  """

  def __init__(self,
               online_pi: snt.Module,
               online_q: snt.Module,
               target_pi: snt.Module,
               target_q: snt.Module,
               num_samples: int,
               online_prior: Optional[snt.Module] = None,
               target_prior: Optional[snt.Module] = None,
               name='OnlineTargetPiQ'):
    super().__init__(name)

    self._online_pi = online_pi
    self._target_pi = target_pi
    self._online_q = online_q
    self._target_q = target_q
    self._online_prior = online_prior
    self._target_prior = target_prior

    self._num_samples = num_samples
    output_list = [
        'online_samples', 'target_samples', 'target_log_probs_behavior_actions',
        'online_log_probs', 'online_q', 'target_q'
    ]
    if online_prior is not None:
      output_list += ['analytic_kl_divergence', 'analytic_kl_to_target']
    self._output_tuple = collections.namedtuple(
        'OnlineTargetPiQ', output_list)

  def __call__(self, input_obs_and_action: Tuple[tf.Tensor, tf.Tensor]):
    (obs, action) = input_obs_and_action
    online_pi_dist = self._online_pi(obs)
    target_pi_dist = self._target_pi(obs)

    online_samples = online_pi_dist.sample(self._num_samples)
    target_samples = target_pi_dist.sample(self._num_samples)
    target_log_probs_behavior_actions = target_pi_dist.log_prob(action)

    online_log_probs = online_pi_dist.log_prob(tf.stop_gradient(online_samples))

    online_q_out = self._online_q(obs, action)
    target_q_out = self._target_q(obs, action)

    output_list = [
        online_samples, target_samples, target_log_probs_behavior_actions,
        online_log_probs, online_q_out, target_q_out
    ]

    if self._online_prior is not None:
      prior_dist = self._online_prior(obs)
      target_prior_dist = self._target_prior(obs)
      analytic_kl_divergence = online_pi_dist.kl_divergence(prior_dist)
      analytic_kl_to_target = online_pi_dist.kl_divergence(target_prior_dist)

      output_list += [analytic_kl_divergence, analytic_kl_to_target]
    output = self._output_tuple(*output_list)
    return output


def static_rnn(core: snt.Module, inputs: types.NestedTensor,
               unroll_length: int):
  """Unroll core along inputs for unroll_length steps.

  Note: for time-major input tensors whose leading dimension is less than
  unroll_length, `None` would be provided instead.

  Args:
    core: an instance of snt.Module.
    inputs: a `nest` of time-major input tensors.
    unroll_length: number of time steps to unroll.

  Returns:
    step_outputs: a `nest` of time-major stacked output tensors of length
      `unroll_length`.
  """
  step_outputs = []
  for time_dim in range(unroll_length):
    inputs_t = tree.map_structure(
        lambda t, i_=time_dim: t[i_] if i_ < t.shape[0] else None, inputs)
    step_output = core(inputs_t)
    step_outputs.append(step_output)

  step_outputs = _nest_stack(step_outputs)
  return step_outputs


def mask_out_restarting(tensor: tf.Tensor, start_of_episode: tf.Tensor):
  """Mask out `tensor` taken on the step that resets the environment.

  Args:
    tensor: a time-major 2-D `Tensor` of shape [T, B].
    start_of_episode: a 2-D `Tensor` of shape [T, B] that contains the points
      where the episode restarts.

  Returns:
    tensor of shape [T, B] with elements are masked out according to step_types,
    restarting weights of shape [T, B]
  """
  tensor.get_shape().assert_has_rank(2)
  start_of_episode.get_shape().assert_has_rank(2)
  weights = tf.cast(~start_of_episode, dtype=tf.float32)
  masked_tensor = tensor * weights
  return masked_tensor


def batch_concat_selection(observation_dict: Dict[str, types.NestedTensor],
                           concat_keys: Optional[Iterable[str]] = None,
                           output_dtype=tf.float32) -> tf.Tensor:
  """Concatenate a dict of observations into 2-D tensors."""
  concat_keys = concat_keys or sorted(observation_dict.keys())
  to_concat = []
  for obs in concat_keys:
    if obs not in observation_dict:
      raise KeyError(
          'Missing observation. Requested: {} (available: {})'.format(
              obs, list(observation_dict.keys())))
    to_concat.append(tf.cast(observation_dict[obs], output_dtype))

  return tf2_utils.batch_concat(to_concat)


def _nest_stack(list_of_nests, axis=0):
  """Convert a list of nests to a nest of stacked lists."""
  return tree.map_structure(lambda *ts: tf.stack(ts, axis=axis), *list_of_nests)
