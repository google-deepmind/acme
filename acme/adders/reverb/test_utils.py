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

"""Utilities for testing Reverb adders."""

from typing import Any, Optional, Sequence, Tuple, TypeVar, Union

from absl.testing import absltest
from acme import specs
from acme.adders import reverb as adders
from acme.adders.reverb import base
from acme.utils import tree_utils
import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree

StepWithExtra = Tuple[Any, dm_env.TimeStep, Any]
StepWithoutExtra = Tuple[Any, dm_env.TimeStep]
Step = TypeVar('Step', StepWithExtra, StepWithoutExtra)


# TODO(b/185308702): consider replacing with a mock-like object.
class FakeWriter(reverb.TrajectoryWriter):
  """Fake writer for testing."""

  def __init__(self, writer: reverb.TrajectoryWriter):
    self._writer = writer
    self.num_episodes = 0

    self.priorities = []
    self.appends = []
    self.closed = False

  @property
  def episode_steps(self):
    return self._writer.episode_steps

  @property
  def history(self) -> Step:
    return self._writer.history

  def append(self, timestep: Step, partial_step: bool = False):
    assert not self.closed, 'Trying to use closed Writer'
    self.appends.append(timestep)
    self._writer.append(timestep, partial_step=partial_step)

  def create_item(self, table: str, priority: float, trajectory: Step):
    assert not self.closed, 'Trying to use closed Writer'
    trajectory_np = tree.map_structure(lambda x: x.numpy(), trajectory)
    self.priorities.append((table, priority, trajectory_np))
    self._writer.create_item(table, priority, trajectory)

  def close(self):
    assert not self.closed, 'Trying to use closed Writer'
    self.closed = True
    self._writer.close()

  def end_episode(self,
                  clear_buffers: bool = False,
                  timeout_ms: Optional[int] = None):
    assert not self.closed, 'Trying to use closed Writer'
    self.num_episodes += 1
    self._writer.end_episode(clear_buffers=clear_buffers, timeout_ms=timeout_ms)

  def flush(self,
            block_until_num_items: int = 0,
            timeout_ms: Optional[int] = None):
    assert not self.closed, 'Trying to use closed Writer'
    self._writer.flush(block_until_num_items=block_until_num_items,
                       timeout_ms=timeout_ms)


class FakeClient(reverb.Client):
  """Fake client for testing."""

  def __init__(self, server_address: str):
    super().__init__(server_address)
    self.writer = None

  def trajectory_writer(self,
                        num_keep_alive_refs: int,
                        get_signature_timeout_ms: Optional[int] = 3000):
    self.writer = FakeWriter(super().trajectory_writer(num_keep_alive_refs))
    return self.writer


def make_trajectory(observations):
  """Make a simple trajectory from a sequence of observations.

  Arguments:
    observations: a sequence of observations.

  Returns:
    a tuple (first, steps) where first contains the initial dm_env.TimeStep
    object and steps contains a list of (action, step) tuples. The length of
    steps is given by episode_length.
  """
  first = dm_env.restart(observations[0])
  middle = [(0, dm_env.transition(reward=0.0, observation=observation))
            for observation in observations[1:-1]]
  last = (0, dm_env.termination(reward=0.0, observation=observations[-1]))
  return first, middle + [last]


def make_sequence(observations):
  """Create a sequence of timesteps of the form `first, [second, ..., last]`."""
  first, steps = make_trajectory(observations)
  observation = first.observation
  sequence = []
  start_of_episode = True
  for action, timestep in steps:
    extras = ()
    sequence.append((observation, action, timestep.reward, timestep.discount,
                     start_of_episode, extras))
    observation = timestep.observation
    start_of_episode = False
  sequence.append((observation, 0, 0.0, 0.0, False, ()))
  return sequence


def _numeric_to_spec(x: Union[float, int, np.ndarray]):
  if isinstance(x, np.ndarray):
    return specs.Array(shape=x.shape, dtype=x.dtype)
  elif isinstance(x, (float, int)):
    return specs.Array(shape=(), dtype=type(x))
  else:
    raise ValueError(f'Unsupported numeric: {type(x)}')


class AdderTestMixin(absltest.TestCase):
  """A helper mixin for testing Reverb adders.

  Note that any test inheriting from this mixin must also inherit from something
  that provides the Python unittest assert methods.
  """

  server: reverb.Server
  client: reverb.Client

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=1000,
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )
    cls.server = reverb.Server([replay_table])

  def setUp(self):
    super().setUp()
    # The adder is used to insert observations into replay.
    address = f'localhost:{self.server.port}'
    self.client = FakeClient(address)

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    cls.server.stop()

  def run_test_adder(
      self,
      adder: base.ReverbAdder,
      first: dm_env.TimeStep,
      steps: Sequence[Step],
      expected_items: Sequence[Any],
      pack_expected_items: bool = False,
      stack_sequence_fields: bool = True,
      repeat_episode_times: int = 1,
      end_behavior: adders.EndBehavior = adders.EndBehavior.ZERO_PAD):
    """Runs a unit test case for the adder.

    Args:
      adder: The instance of `base.ReverbAdder` that is being tested.
      first: The first `dm_env.TimeStep` that is used to call
        `base.ReverbAdder.add_first()`.
      steps: A sequence of (action, timestep) tuples that are passed to
        `base.ReverbAdder.add()`.
      expected_items: The sequence of items that are expected to be created
        by calling the adder's `add_first()` method on `first` and `add()` on
        all of the elements in `steps`.
      pack_expected_items: Deprecated and not used. If true the expected items
        are given unpacked and need to be packed in a list before comparison.
      stack_sequence_fields: Whether to stack the sequence fields of the
        expected items before comparing to the observed items. Usually False
        for transition adders and True for both episode and sequence adders.
      repeat_episode_times: How many times to run an episode.
      end_behavior: How end of episode should be handled.
    """

    del pack_expected_items

    if not steps:
      raise ValueError('At least one step must be given.')

    has_extras = len(steps[0]) == 3
    env_spec = tree.map_structure(
        _numeric_to_spec,
        specs.EnvironmentSpec(
            observations=steps[0][1].observation,
            actions=steps[0][0],
            rewards=steps[0][1].reward,
            discounts=steps[0][1].discount))
    if has_extras:
      extras_spec = tree.map_structure(_numeric_to_spec, steps[0][2])
    else:
      extras_spec = ()
    signature = adder.signature(env_spec, extras_spec=extras_spec)

    for episode_id in range(repeat_episode_times):
      # Add all the data up to the final step.
      adder.add_first(first)
      for step in steps[:-1]:
        action, ts = step[0], step[1]

        if has_extras:
          extras = step[2]
        else:
          extras = ()

        adder.add(action, next_timestep=ts, extras=extras)

      # Add the final step.
      adder.add(*steps[-1])

    # Ending the episode should close the writer. No new writer should yet have
    # been created as it is constructed lazily.
    if end_behavior is not adders.EndBehavior.CONTINUE:
      self.assertEqual(self.client.writer.num_episodes, repeat_episode_times)

    # Make sure our expected and observed data match.
    observed_items = [p[2] for p in self.client.writer.priorities]

    # Check matching number of items.
    self.assertEqual(len(expected_items), len(observed_items))

    # Check items are matching according to numpy's almost_equal.
    for expected_item, observed_item in zip(expected_items, observed_items):
      if stack_sequence_fields:
        expected_item = tree_utils.stack_sequence_fields(expected_item)

      # Set check_types=False because we check them below.
      tree.map_structure(
          np.testing.assert_array_almost_equal,
          expected_item,
          tuple(observed_item),
          check_types=False)

    # Make sure the signature matches was is being written by Reverb.
    def _check_signature(spec: tf.TensorSpec, value: np.ndarray):
      self.assertTrue(spec.is_compatible_with(tf.convert_to_tensor(value)))

    # Check the last transition's signature.
    tree.map_structure(_check_signature, signature, observed_items[-1])
