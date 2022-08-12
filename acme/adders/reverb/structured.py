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

"""Generic adders that wraps Reverb's StructuredWriter."""

import itertools
import time

from typing import Callable, List, Optional, Sequence, Sized

from absl import logging
from acme import specs
from acme import types
from acme.adders import base as adders_base
from acme.adders.reverb import base as reverb_base
from acme.adders.reverb import sequence as sequence_adder
import dm_env
import numpy as np
import reverb
from reverb import structured_writer as sw
import tensorflow as tf
import tree

Step = reverb_base.Step
Trajectory = reverb_base.Trajectory
EndBehavior = sequence_adder.EndBehavior

_RESET_WRITER_EVERY_SECONDS = 60


class StructuredAdder(adders_base.Adder):
  """Generic Adder which writes to Reverb using Reverb's `StructuredWriter`.

  The StructuredAdder is a thin wrapper around Reverb's `StructuredWriter` and
  its behaviour is determined by the configs to __init__. Much of the behaviour
  provided by other Adders can be replicated using `StructuredAdder` but there
  are a few noteworthy differences:

   * The behaviour of `StructuredAdder` can be thought of as the union of all
     its configs. This means that a single adder is capable of inserting items
     of different structures into any number of tables WITHOUT any data
     duplication. Other adders are only capable of writing items of the same
     structure into multiple tables.
   * The complete structure of the step must be known at construction time when
     using the StructuredAdder. This is not the case for other Adders as they
     allow the structure of the step to become expanded over time.
   * The `StructuredAdder` assigns all items the same priority (1.0) as it does
     not currently support custom priority computations.
   * The StructuredAdder is completely generic and thus does not perform any
     preprocessing on the data (e.g. cumulative rewards as done by the
     NStepTransitionAdder) before writing it to Reverb. The user is instead
     expected to perform preprocessing in the dataset pipeline on the learner.
  """

  def __init__(self, client: reverb.Client, max_in_flight_items: int,
               configs: Sequence[sw.Config], step_spec: Step):
    """Initialize a StructuredAdder instance.

    Args:
      client: A client to the Reverb backend.
      max_in_flight_items: The maximum number of items allowed to be "in flight"
        at the same time. See `block_until_num_items` in
        `reverb.TrajectoryWriter.flush` for more info.
      configs: Configurations defining the behaviour of the wrapped Reverb
        writer.
      step_spec: spec of the step that is going to be inserted in the Adder. It
        can be created with `create_step_spec` using the environment spec and
        and the extras spec.
    """

    # We validate the configs by attempting to infer the signatures of all
    # targeted tables.
    for table, table_configs in itertools.groupby(configs, lambda c: c.table):
      try:
        sw.infer_signature(list(table_configs), step_spec)
      except ValueError as e:
        raise ValueError(
            f'Received invalid configs for table {table}: {str(e)}') from e

    self._client = client
    self._configs = tuple(configs)
    self._none_step: Step = tree.map_structure(lambda _: None, step_spec)
    self._max_in_flight_items = max_in_flight_items

    self._writer = None
    self._writer_created_at = None

  def __del__(self):
    if self._writer is None:
      return

    # Try flush all appended data before closing to avoid loss of experience.
    try:
      self._writer.flush(0, timeout_ms=10_000)
    except reverb.DeadlineExceededError as e:
      logging.error(
          'Timeout (10 s) exceeded when flushing the writer before '
          'deleting it. Caught Reverb exception: %s', str(e))

  def _make_step(self, **kwargs) -> Step:
    """Complete the step with None in the missing positions."""
    return Step(**{**self._none_step._asdict(), **kwargs})

  @property
  def configs(self):
    return self._configs

  def reset(self, timeout_ms: Optional[int] = None):
    """Marks the active episode as completed and flushes pending items."""
    if self._writer is not None:
      # Flush all pending items.
      self._writer.end_episode(clear_buffers=True, timeout_ms=timeout_ms)

      # Create a new writer unless the current one is too young.
      # This is to reduce the relative overhead of creating a new Reverb writer.
      if time.time() - self._writer_created_at > _RESET_WRITER_EVERY_SECONDS:
        self._writer = None

  def add_first(self, timestep: dm_env.TimeStep):
    """Record the first observation of an episode."""
    if not timestep.first():
      raise ValueError(
          'adder.add_first called with a timestep that was not the first of its'
          'episode (i.e. one for which timestep.first() is not True)')

    if self._writer is None:
      self._writer = self._client.structured_writer(self._configs)
      self._writer_created_at = time.time()

    # Record the next observation but leave the history buffer row open by
    # passing `partial_step=True`.
    self._writer.append(
        data=self._make_step(
            observation=timestep.observation,
            start_of_episode=timestep.first()),
        partial_step=True)
    self._writer.flush(self._max_in_flight_items)

  def add(self,
          action: types.NestedArray,
          next_timestep: dm_env.TimeStep,
          extras: types.NestedArray = ()):
    """Record an action and the following timestep."""

    if not self._writer.step_is_open:
      raise ValueError('adder.add_first must be called before adder.add.')

    # Add the timestep to the buffer.
    has_extras = (
        len(extras) > 0 if isinstance(extras, Sized)  # pylint: disable=g-explicit-length-test
        else extras is not None)

    current_step = self._make_step(
        action=action,
        reward=next_timestep.reward,
        discount=next_timestep.discount,
        extras=extras if has_extras else self._none_step.extras)
    self._writer.append(current_step)

    # Record the next observation and write.
    self._writer.append(
        data=self._make_step(
            observation=next_timestep.observation,
            start_of_episode=next_timestep.first()),
        partial_step=True)
    self._writer.flush(self._max_in_flight_items)

    if next_timestep.last():
      # Complete the row by appending zeros to remaining open fields.
      # TODO(b/183945808): remove this when fields are no longer expected to be
      # of equal length on the learner side.
      dummy_step = tree.map_structure(
          lambda x: None if x is None else np.zeros_like(x), current_step)
      self._writer.append(dummy_step)
      self.reset()


def create_step_spec(
    environment_spec: specs.EnvironmentSpec, extras_spec: types.NestedSpec = ()
) -> Step:
  return Step(
      *environment_spec,
      start_of_episode=tf.TensorSpec([], tf.bool, 'start_of_episode'),
      extras=extras_spec)


def _last_n(n: int, step_spec: Step) -> Trajectory:
  """Constructs a sequence with the last n elements of all the Step fields."""
  return Trajectory(*tree.map_structure(lambda x: x[-n:], step_spec))


def create_sequence_config(
    step_spec: Step,
    sequence_length: int,
    period: int,
    table: str = reverb_base.DEFAULT_PRIORITY_TABLE,
    end_of_episode_behavior: EndBehavior = EndBehavior.TRUNCATE,
    sequence_pattern: Callable[[int, Step], Trajectory] = _last_n,
) -> List[sw.Config]:
  """Generates configs that produces the same behaviour as `SequenceAdder`.

  NOTE! ZERO_PAD is not supported as the same behaviour can be achieved by
  writing with TRUNCATE and then adding padding in the dataset pipeline on the
  learner.

  Args:
    step_spec: The full structure of the data which will be appended to the
      Reverb `StructuredWriter` in each step. Please use `create_step_spec` to
      create `step_spec`.
    sequence_length: The number of steps that each trajectory should span.
    period: The period with which we add sequences. If less than
      sequence_length, overlapping sequences are added. If equal to
      sequence_length, sequences are exactly non-overlapping.
    table: Name of the Reverb table to write items to. Defaults to the default
      Acme table.
    end_of_episode_behavior: Determines how sequences at the end of the episode
      are handled (default `EndOfEpisodeBehavior.TRUNCATE`). See the docstring
      of `EndOfEpisodeBehavior` for more information.
    sequence_pattern: Transformation to obtain a sequence given the length
      and the shape of the step.

  Returns:
    A list of configs for `StructuredAdder` to produce the described behaviour.

  Raises:
    ValueError: If sequence_length is <= 0.
    NotImplementedError: If `end_of_episod_behavior` is `ZERO_PAD`.
  """
  if sequence_length <= 0:
    raise ValueError(f'sequence_length must be > 0 but got {sequence_length}.')

  if end_of_episode_behavior == EndBehavior.ZERO_PAD:
    raise NotImplementedError(
        'Zero-padding is not supported. Please use TRUNCATE instead.')

  if end_of_episode_behavior == EndBehavior.CONTINUE:
    raise NotImplementedError('Merging episodes is not supported.')

  def _sequence_pattern(n: int) -> sw.Pattern:
    return sw.pattern_from_transform(step_spec,
                                     lambda step: sequence_pattern(n, step))

  # The base config is considered for all but the last step in the episode. No
  # trajectories are created for the first `sequence_step-1` steps and then a
  # new trajectory is inserted every `period` steps.
  base_config = sw.create_config(
      pattern=_sequence_pattern(sequence_length),
      table=table,
      conditions=[
          sw.Condition.step_index() >= sequence_length - 1,
          sw.Condition.step_index() % period == (sequence_length - 1) % period,
      ])

  end_of_episode_configs = []
  if end_of_episode_behavior == EndBehavior.WRITE:
    # Simply write a trajectory in exactly the same way as the base config. The
    # only difference here is that we ALWAYS create a trajectory even if it
    # doesn't align with the `period`. The exceptions to the rule are episodes
    # that are shorter than `sequence_length` steps which are completely
    # ignored.
    config = sw.create_config(
        pattern=_sequence_pattern(sequence_length),
        table=table,
        conditions=[
            sw.Condition.is_end_episode(),
            sw.Condition.step_index() >= sequence_length - 1,
        ])
    end_of_episode_configs.append(config)
  elif end_of_episode_behavior == EndBehavior.TRUNCATE:
    # The first trajectory is written at step index `sequence_length - 1` and
    # then written every `period` step. This means that the
    # `step_index % period` will always be equal to the below value everytime a
    # trajectory is written.
    target = (sequence_length - 1) % period

    # When the episode ends we still want to capture the steps that has been
    # appended since the last item was created. We do this by creating a config
    # for all `step_index % period`, except `target`, and condition these
    # configs so that they only are triggered when `end_episode` is called.
    for x in range(period):
      # When the last step is aligned with the period of the inserts then no
      # action is required as the item was already generated by `base_config`.
      if x == target:
        continue

      # If we were to pad the trajectory then we'll need to continue adding
      # padding until `step_index % period` is equal to `target` again. We can
      # exploit this relation by conditioning the config to only be applied for
      # a single value of `step_index % period`. This constraint means that we
      # can infer the number of padding steps required until the next write
      # would have occurred if the episode didn't end.
      #
      # Now if we assume that the padding instead is added on the dataset (or
      # the trajectory is simply truncated) then we can infer from the above
      # that the number of real steps in this padded trajectory will be the
      # difference between `sequence_length` and number of pad steps.
      num_pad_steps = (target - x) % period
      unpadded_length = sequence_length - num_pad_steps

      config = sw.create_config(
          pattern=_sequence_pattern(unpadded_length),
          table=table,
          conditions=[
              sw.Condition.is_end_episode(),
              sw.Condition.step_index() % period == x,
              sw.Condition.step_index() >= sequence_length,
          ])
      end_of_episode_configs.append(config)

    # The above configs will capture the "remainder" of any episode that is at
    # least `sequence_length` steps long. However, if the entire episode is
    # shorter than `sequence_length` then data might still be lost. We avoid
    # this by simply creating `sequence_length-1` configs that capture the last
    # `x` steps iff the entire episode is `x` steps long.
    for x in range(1, sequence_length):
      config = sw.create_config(
          pattern=_sequence_pattern(x),
          table=table,
          conditions=[
              sw.Condition.is_end_episode(),
              sw.Condition.step_index() == x - 1,
          ])
      end_of_episode_configs.append(config)
  else:
    raise ValueError(
        f'Unexpected `end_of_episod_behavior`: {end_of_episode_behavior}')

  return [base_config] + end_of_episode_configs


def create_n_step_transition_config(
    step_spec: Step,
    n_step: int,
    table: str = reverb_base.DEFAULT_PRIORITY_TABLE) -> List[sw.Config]:
  """Generates configs that replicates the behaviour of NStepTransitionAdder.

  Please see the docstring of NStepTransitionAdder for more details.

  NOTE! In contrast to NStepTransitionAdder, the trajectories written by the
  `StructuredWriter` does not include the precomputed cumulative reward and
  discounts. Instead the trajectory includes the raw rewards and discounts
  required to comptute these values.

  Args:
    step_spec: The full structure of the data which will be appended to the
      Reverb `StructuredWriter` in each step. Please use `create_step_spec` to
      create `step_spec`.
    n_step: The "N" in N-step transition. See the class docstring for the
      precise definition of what an N-step transition is. `n_step` must be at
      least 1, in which case we use the standard one-step transition, i.e. (s_t,
      a_t, r_t, d_t, s_t+1, e_t).
    table: Name of the Reverb table to write items to. Defaults to the default
      Acme table.

  Returns:
    A list of configs for `StructuredAdder` to produce the described behaviour.
  """

  def _make_pattern(n: int):
    ref_step = sw.create_reference_step(step_spec)

    get_first = lambda x: x[-(n + 1):-n]
    get_all = lambda x: x[-(n + 1):-1]
    get_first_and_last = lambda x: x[-(n + 1)::n]

    tmap = tree.map_structure

    # We use the exact same structure as we done when writing sequences except
    # we trim the number of steps in each sub tree. This has the benefit that
    # the postprocessing used to transform these items into N-step transition
    # structures (cumulative rewards and discounts etc.) can be applied on
    # full sequence items as well. The only difference being that the latter is
    # more wasteful than the trimmed down version we write here.
    return Trajectory(
        observation=tmap(get_first_and_last, ref_step.observation),
        action=tmap(get_first, ref_step.action),
        reward=tmap(get_all, ref_step.reward),
        discount=tmap(get_all, ref_step.discount),
        start_of_episode=tmap(get_first, ref_step.start_of_episode),
        extras=tmap(get_first, ref_step.extras))

  # At the start of the episodes we'll add shorter transitions.
  start_of_episode_configs = []
  for n in range(1, n_step):
    config = sw.create_config(
        pattern=_make_pattern(n),
        table=table,
        conditions=[
            sw.Condition.step_index() == n,
        ],
    )
    start_of_episode_configs.append(config)

  # During all other steps we'll add a full N-step transition.
  base_config = sw.create_config(pattern=_make_pattern(n_step), table=table)

  # When the episode ends we'll add shorter transitions.
  end_of_episode_configs = []
  for n in range(n_step - 1, 0, -1):
    config = sw.create_config(
        pattern=_make_pattern(n),
        table=table,
        conditions=[
            sw.Condition.is_end_episode(),
            # If the entire episode is shorter than n_step then the episode
            # start configs will already create an item that covers all the
            # steps so we add this filter here to avoid adding it again.
            sw.Condition.step_index() != n,
        ],
    )
    end_of_episode_configs.append(config)

  return start_of_episode_configs + [base_config] + end_of_episode_configs
