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

"""An adder useful in the context of Learning From Demonstrations.

This adder is mixing the collected episodes with some demonstrations
coming from an offline dataset.

TODO(damienv): Mixing demonstrations and collected episodes could also be
  done when reading from the replay buffer. In that case, all the processing
  applied by reverb should also be applied on the demonstrations.
  Design wise, both solutions make equally sense. The alternative solution
  could then be later implemented as well.
"""

from typing import Any, Iterator, Tuple

from acme import adders
from acme import types
import dm_env


class LfdAdder(adders.Adder):
  """Adder which adds from time to time some demonstrations.

  Lfd stands for Learning From Demonstrations and is the same technique
  as the one used in R2D3.
  """

  def __init__(self,
               adder: adders.Adder,
               demonstrations: Iterator[Tuple[Any, dm_env.TimeStep]],
               initial_insert_count: int,
               demonstration_ratio: float):
    """LfdAdder constructor.

    Args:
      adder: The underlying adder used to add mixed episodes.
      demonstrations: An iterator on infinite stream of (action, next_timestep)
        pairs. Episode boundaries are defined by TimeStep.FIRST and
        timestep.LAST markers. Note that the first action of an episode is
        ignored. Note also that proper uniform sampling of demonstrations is the
        responsibility of the iterator.
      initial_insert_count: Number of steps of demonstrations to add before
        adding any step of the collected episodes. Note that since only full
        episodes can be added, this number of steps is only a target.
      demonstration_ratio: Ratio of demonstration steps to add to the underlying
        adder. ratio = num_demonstration_steps_added / total_num_steps_added
        and must be in [0, 1).
        Note that this ratio is the desired ratio in the steady behavior
        and does not account for the initial inserts of demonstrations.
        Note also that this ratio is only a target ratio since the granularity
        is the episode.
    """
    self._adder = adder
    self._demonstrations = demonstrations
    self._demonstration_ratio = demonstration_ratio
    if demonstration_ratio < 0 or demonstration_ratio >= 1.:
      raise ValueError('Invalid demonstration ratio.')

    # Number of demonstration steps that should have been added to the replay
    # buffer to meet the target demonstration ratio minus what has been really
    # added.
    # As a consequence:
    # - when this delta is zero, the effective ratio exactly matches the desired
    #   ratio
    # - when it is positive, more demonstrations need to be added to
    #   reestablish the balance
    # The initial value is set so that after exactly initial_insert_count
    # inserts of demonstration steps, _delta_demonstration_step_count will be
    # zero.
    self._delta_demonstration_step_count = (
        (1. - self._demonstration_ratio) * initial_insert_count)

  def reset(self):
    self._adder.reset()

  def _add_demonstration_episode(self):
    _, timestep = next(self._demonstrations)
    if not timestep.first():
      raise ValueError('Expecting the start of an episode.')
    self._adder.add_first(timestep)
    self._delta_demonstration_step_count -= (1. - self._demonstration_ratio)
    while not timestep.last():
      action, timestep = next(self._demonstrations)
      self._adder.add(action, timestep)
      self._delta_demonstration_step_count -= (1. - self._demonstration_ratio)

    # Reset is being called periodically to reset the connection to reverb.
    # TODO(damienv, bshahr): Make the reset an internal detail of the reverb
    # adder and remove it from the adder API.
    self._adder.reset()

  def add_first(self, timestep: dm_env.TimeStep):
    while self._delta_demonstration_step_count > 0.:
      self._add_demonstration_episode()

    self._adder.add_first(timestep)
    self._delta_demonstration_step_count += self._demonstration_ratio

  def add(self,
          action: types.NestedArray,
          next_timestep: dm_env.TimeStep,
          extras: types.NestedArray = ()):
    self._adder.add(action, next_timestep)
    self._delta_demonstration_step_count += self._demonstration_ratio
