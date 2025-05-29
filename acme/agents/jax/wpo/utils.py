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

"""Utility functions for WPO agent."""

from acme import types
from acme.adders import reverb as adders
import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


def _fetch_devicearray(x):
  if isinstance(x, jax.Array):
    return np.asarray(x)
  return x


def get_from_first_device(nest, as_numpy: bool = True):
  """Gets the first array of a nest of `jax.pxla.ShardedDeviceArray`s."""
  # TODO(abef): remove this when fake_pmap is fixed or acme error is removed.

  def _slice_and_maybe_to_numpy(x):
    x = x[0]
    return _fetch_devicearray(x) if as_numpy else x

  return jax.tree.map(_slice_and_maybe_to_numpy, nest)


def make_sequences_from_transitions(
    transitions: types.Transition,
    num_batch_dims: int = 1) -> adders.Step:
  """Convert a batch of transitions into a batch of 1-step sequences."""
  stack = lambda x, y: jnp.stack((x, y), axis=num_batch_dims)
  duplicate = lambda x: stack(x, x)
  observation = jax.tree.map(
      stack, transitions.observation, transitions.next_observation
  )
  reward = duplicate(transitions.reward)

  return adders.Step(  # pytype: disable=wrong-arg-types  # jnp-type
      observation=observation,
      action=duplicate(transitions.action),
      reward=reward,
      discount=duplicate(transitions.discount),
      start_of_episode=jnp.zeros_like(reward, dtype=jnp.bool_),
      extras=jax.tree.map(duplicate, transitions.extras),
  )
