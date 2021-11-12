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

"""Episode adders.

This implements full episode adders, potentially with padding.
"""

from typing import Callable, Optional, Iterable, Tuple

from acme import specs
from acme import types
from acme.adders.reverb import base
from acme.adders.reverb import utils

import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree

_PaddingFn = Callable[[Tuple[int, ...], np.dtype], np.ndarray]


class EpisodeAdder(base.ReverbAdder):
  """Adder which adds entire episodes as trajectories."""

  def __init__(
      self,
      client: reverb.Client,
      max_sequence_length: int,
      delta_encoded: bool = False,
      priority_fns: Optional[base.PriorityFnMapping] = None,
      max_in_flight_items: int = 1,
      padding_fn: Optional[_PaddingFn] = None,
      # Deprecated kwargs.
      chunk_length: Optional[int] = None,
  ):
    del chunk_length

    super().__init__(
        client=client,
        max_sequence_length=max_sequence_length,
        delta_encoded=delta_encoded,
        priority_fns=priority_fns,
        max_in_flight_items=max_in_flight_items,
    )
    self._padding_fn = padding_fn

  def add(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
      extras: types.NestedArray = (),
  ):
    if self._writer.episode_steps >= self._max_sequence_length - 1:
      raise ValueError(
          'The number of observations within the same episode will exceed '
          'max_sequence_length with the addition of this transition.')

    super().add(action, next_timestep, extras)

  def _write(self):
    # This adder only writes at the end of the episode, see _write_last()
    pass

  def _write_last(self):
    if self._padding_fn is not None and self._writer.episode_steps < self._max_sequence_length:
      history = self._writer.history
      padding_step = dict(
          observation=history['observation'],
          action=history['action'],
          reward=history['reward'],
          discount=history['discount'],
          extras=history.get('extras', ()))
      # Get shapes and dtypes from the last element.
      padding_step = tree.map_structure(
          lambda col: self._padding_fn(col[-1].shape, col[-1].dtype),
          padding_step)
      padding_step['start_of_episode'] = False
      while self._writer.episode_steps < self._max_sequence_length:
        self._writer.append(padding_step)

    trajectory = tree.map_structure(lambda x: x[:], self._writer.history)

    # Pack the history into a base.Step structure and get numpy converted
    # variant for priotiy computation.
    trajectory = base.Trajectory(**trajectory)

    # Calculate the priority for this episode.
    table_priorities = utils.calculate_priorities(self._priority_fns,
                                                  trajectory)

    # Create a prioritized item for each table.
    for table_name, priority in table_priorities.items():
      self._writer.create_item(table_name, priority, trajectory)
      self._writer.flush(self._max_in_flight_items)

  # TODO(b/185309817): make this into a standalone method.
  @classmethod
  def signature(cls,
                environment_spec: specs.EnvironmentSpec,
                extras_spec: types.NestedSpec = (),
                sequence_length: Optional[int] = None):
    """This is a helper method for generating signatures for Reverb tables.

    Signatures are useful for validating data types and shapes, see Reverb's
    documentation for details on how they are used.

    Args:
      environment_spec: A `specs.EnvironmentSpec` whose fields are nested
        structures with leaf nodes that have `.shape` and `.dtype` attributes.
        This should come from the environment that will be used to generate the
        data inserted into the Reverb table.
      extras_spec: A nested structure with leaf nodes that have `.shape` and
        `.dtype` attributes. The structure (and shapes/dtypes) of this must be
        the same as the `extras` passed into `ReverbAdder.add`.
      sequence_length: An optional integer representing the expected length of
        sequences that will be added to replay.

    Returns:
      A `Step` whose leaf nodes are `tf.TensorSpec` objects.
    """

    def add_time_dim(paths: Iterable[str], spec: tf.TensorSpec):
      return tf.TensorSpec(
          shape=(sequence_length, *spec.shape),
          dtype=spec.dtype,
          name='/'.join(str(p) for p in paths))

    trajectory_env_spec, trajectory_extras_spec = tree.map_structure_with_path(
        add_time_dim, (environment_spec, extras_spec))

    trajectory_spec = base.Trajectory(
        *trajectory_env_spec,
        start_of_episode=tf.TensorSpec(
            shape=(sequence_length,), dtype=tf.bool, name='start_of_episode'),
        extras=trajectory_extras_spec)

    return trajectory_spec
