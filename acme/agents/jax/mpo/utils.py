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

"""Utility functions for MPO agent."""

from typing import Callable

from acme import types
from acme.adders import reverb as adders
from acme.agents.jax.mpo import types as mpo_types
import distrax
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


def rolling_window(x: jnp.ndarray,
                   window: int,
                   axis: int = 0,
                   time_major: bool = True):
  """Stack the N=T-W+1 length W slices [0:W, 1:W+1, ..., T-W:T] from a tensor.

  Args:
    x: The tensor to select rolling slices from (along specified axis), with
      shape [..., T, ...]; i.e., T = x.shape[axis].
    window: The length (W) of the slices to select.
    axis: The axis to slice from (defaults to 0).
    time_major: If true, output will have shape [..., W, N, ...], otherwise
      it will have shape [..., N, W, ...], where x.shape is [..., T, ...].

  Returns:
    A tensor containing the stacked slices [0:W, ... T-W:T] from an axis of x.
  """
  sequence_length = x.shape[axis]
  starts = jnp.arange(sequence_length - window + 1)
  ends = jnp.arange(window)
  if time_major:
    idx = starts[None, :] + ends[:, None]  # Output will be [..., W, N, ...].
  else:
    idx = starts[:, None] + ends[None, :]  # Output will be [..., N, W, ...].
  out = jnp.take(x, idx, axis=axis)
  return out


def tree_map_distribution(
    f: Callable[[mpo_types.DistributionOrArray], mpo_types.DistributionOrArray],
    x: mpo_types.DistributionOrArray) -> mpo_types.DistributionOrArray:
  """Apply a jax function to a distribution by treating it as tree."""
  if isinstance(x, distrax.Distribution):
    safe_f = lambda y: f(y) if isinstance(y, jnp.ndarray) else y
    nil, tree_data = x.tree_flatten()
    new_tree_data = jax.tree.map(safe_f, tree_data)
    new_x = x.tree_unflatten(new_tree_data, nil)
    return new_x
  elif isinstance(x, tfd.Distribution):
    return jax.tree.map(f, x)
  else:
    return f(x)


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
