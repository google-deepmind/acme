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

"""Multiplexers are networks that take multiple inputs."""

from typing import Callable, Optional, Union

from acme import types
from acme.utils import tf2_utils

import sonnet as snt
import tensorflow as tf

TensorTransformation = Union[snt.Module, Callable[[types.NestedTensor],
                                                  tf.Tensor]]


class CriticMultiplexer(snt.Module):
  """Module connecting a critic torso to (transformed) observations/actions.

  This takes as input a `critic_network`, an `observation_network`, and an
  `action_network` and returns another network whose outputs are given by
  `critic_network(observation_network(o), action_network(a))`.

  The observations and actions passed to this module are assumed to have a batch
  dimension that match.

  Notes:
  - Either the `observation_` or `action_network` can be `None`, in which case
    the observation or action, resp., are passed to the critic network as is.
  - If all `critic_`, `observation_` and `action_network` are `None`, this
    module reduces to a simple `tf2_utils.batch_concat()`.
  """

  def __init__(self,
               critic_network: Optional[TensorTransformation] = None,
               observation_network: Optional[TensorTransformation] = None,
               action_network: Optional[TensorTransformation] = None):
    self._critic_network = critic_network
    self._observation_network = observation_network
    self._action_network = action_network
    super().__init__(name='critic_multiplexer')

  def __call__(self, observation: tf.Tensor, action: tf.Tensor) -> tf.Tensor:

    # Maybe transform observations and actions before feeding them on.
    if self._observation_network:
      observation = self._observation_network(observation)
    if self._action_network:
      action = self._action_network(action)

    # Concat observations and actions, with one batch dimension.
    outputs = tf2_utils.batch_concat([observation, action])

    # Maybe transform output before returning.
    if self._critic_network:
      outputs = self._critic_network(outputs)

    return outputs
