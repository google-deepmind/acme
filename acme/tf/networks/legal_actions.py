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

"""Networks used for handling illegal actions."""

from typing import Any, Callable, Iterable, Optional, Union

# pytype: disable=import-error
from acme.wrappers import open_spiel_wrapper
# pytype: enable=import-error

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class MaskedSequential(snt.Module):
  """Applies a legal actions mask to a linear chain of modules / callables.

  It is assumed the trailing dimension of the final layer (representing
  action values) is the same as the trailing dimension of legal_actions.
  """

  def __init__(self,
               layers: Optional[Iterable[Callable[..., Any]]] = None,
               name: str = 'MaskedSequential'):
    super().__init__(name=name)
    self._layers = list(layers) if layers is not None else []
    self._illegal_action_penalty = -1e9
    # Note: illegal_action_penalty cannot be -np.inf because trfl's qlearning
    # ops utilize a batched_index function that returns NaN whenever -np.inf
    # is present among action values.

  def __call__(self, inputs: open_spiel_wrapper.OLT) -> tf.Tensor:
    # Extract observation, legal actions, and terminal
    outputs = inputs.observation
    legal_actions = inputs.legal_actions
    terminal = inputs.terminal

    for mod in self._layers:
      outputs = mod(outputs)

    # Apply legal actions mask
    outputs = tf.where(tf.equal(legal_actions, 1), outputs,
                       tf.fill(tf.shape(outputs), self._illegal_action_penalty))

    # When computing the Q-learning target (r_t + d_t * max q_t) we need to
    # ensure max q_t = 0 in terminal states.
    outputs = tf.where(tf.equal(terminal, 1), tf.zeros_like(outputs), outputs)

    return outputs


# FIXME: Add functionality to support decaying epsilon parameter.
# FIXME: This is a modified version of trfl's epsilon_greedy() which
# incorporates code from the bug fix described here
# https://github.com/deepmind/trfl/pull/28
class EpsilonGreedy(snt.Module):
  """Computes an epsilon-greedy distribution over actions.

  This policy does the following:
  - With probability 1 - epsilon, take the action corresponding to the highest
  action value, breaking ties uniformly at random.
  - With probability epsilon, take an action uniformly at random.
  """

  def __init__(self,
               epsilon: Union[tf.Tensor, float],
               threshold: float,
               name: str = 'EpsilonGreedy'):
    """Initialize the policy.

    Args:
      epsilon: Exploratory param with value between 0 and 1.
      threshold: Action values must exceed this value to qualify as a legal
        action and possibly be selected by the policy.
      name: Name of the network.

    Returns:
      policy: tfp.distributions.Categorical distribution representing the
        policy.
    """
    super().__init__(name=name)
    self._epsilon = tf.Variable(epsilon, trainable=False)
    self._threshold = threshold

  def __call__(self, action_values: tf.Tensor) -> tfd.Categorical:
    legal_actions_mask = tf.where(
        tf.math.less_equal(action_values, self._threshold),
        tf.fill(tf.shape(action_values), 0.),
        tf.fill(tf.shape(action_values), 1.))

    # Dithering action distribution.
    dither_probs = 1 / tf.reduce_sum(legal_actions_mask, axis=-1,
                                     keepdims=True) * legal_actions_mask
    masked_action_values = tf.where(tf.equal(legal_actions_mask, 1),
                                    action_values,
                                    tf.fill(tf.shape(action_values), -np.inf))
    # Greedy action distribution, breaking ties uniformly at random.
    max_value = tf.reduce_max(masked_action_values, axis=-1, keepdims=True)
    greedy_probs = tf.cast(
        tf.equal(action_values * legal_actions_mask, max_value),
        action_values.dtype)

    greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)

    # Epsilon-greedy action distribution.
    probs = self._epsilon * dither_probs + (1 - self._epsilon) * greedy_probs

    # Make the policy object.
    policy = tfd.Categorical(probs=probs)

    return policy
