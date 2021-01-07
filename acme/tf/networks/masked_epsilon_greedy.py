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

"""Wrapping trfl epsilon_greedy with legal action masking."""

from typing import Optional, Mapping, Union

import sonnet as snt
import tensorflow as tf
import trfl


class NetworkWithMaskedEpsilonGreedy(snt.Module):
  """Epsilon greedy sampling with action masking on network outputs."""

  def __init__(self,
               network: snt.Module,
               epsilon: Optional[tf.Tensor] = None):
    """Initialize the network and epsilon.

    Usage:
      Wrap an observation in a dictionary in your environment as follows:

        observation <-- {"your_key_for_observation": observation,
                         "legal_actions_mask": your_action_mask_tensor}

    and update your network to use 'observation["your_key_for_observation"]'
    rather than 'observation'.

    Args:
      network: the online Q network (the one being optimized)
      epsilon: probability of taking a random action.
    """
    super().__init__()
    self._network = network
    self._epsilon = epsilon

  def __call__(
      self, observation: Union[Mapping[str, tf.Tensor],
                               tf.Tensor]) -> tf.Tensor:
    q = self._network(observation)
    return trfl.epsilon_greedy(
        q, epsilon=self._epsilon,
        legal_actions_mask=observation['legal_actions_mask']).sample()
