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

from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

from absl.testing import absltest
from acme import specs
from acme import types
from acme.adders import base as adders_base
from acme.adders import reverb as adders
from acme.utils import tree_utils
import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree

StepWithExtra = Tuple[Any, dm_env.TimeStep, Any]
StepWithoutExtra = Tuple[Any, dm_env.TimeStep]
Step = TypeVar('Step', StepWithExtra, StepWithoutExtra)


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


def get_specs(step):
  """Infer spec from an example step."""
  env_spec = tree.map_structure(
      _numeric_to_spec,
      specs.EnvironmentSpec(
          observations=step[1].observation,
          actions=step[0],
          rewards=step[1].reward,
          discounts=step[1].discount))

  has_extras = len(step) == 3
  if has_extras:
    extras_spec = tree.map_structure(_numeric_to_spec, step[2])
  else:
    extras_spec = ()

  return env_spec, extras_spec


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

    replay_table = reverb.Table.queue(adders.DEFAULT_PRIORITY_TABLE, 1000)
    cls.server = reverb.Server([replay_table])
    cls.client = reverb.Client(f'localhost:{cls.server.port}')

  def tearDown(self):
    self.client.reset(adders.DEFAULT_PRIORITY_TABLE)
    super().tearDown()

  @classmethod
  def tearDownClass(cls):
    cls.server.stop()
    super().tearDownClass()

  def num_episodes(self):
    info = self.client.server_info(1)[adders.DEFAULT_PRIORITY_TABLE]
    return info.num_episodes

  def num_items(self):
    info = self.client.server_info(1)[adders.DEFAULT_PRIORITY_TABLE]
    return info.current_size

  def items(self):
    sampler = self.client.sample(
        table=adders.DEFAULT_PRIORITY_TABLE,
        num_samples=self.num_items(),
        emit_timesteps=False)
    return [sample.data for sample in sampler]  # pytype: disable=attribute-error

  def run_test_adder(
      self,
      adder: adders_base.Adder,
      first: dm_env.TimeStep,
      steps: Sequence[Step],
      expected_items: Sequence[Any],
      signature: types.NestedSpec,
      pack_expected_items: bool = False,
      stack_sequence_fields: bool = True,
      repeat_episode_times: int = 1,
      end_behavior: adders.EndBehavior = adders.EndBehavior.ZERO_PAD,
      item_transform: Optional[Callable[[Sequence[np.ndarray]], Any]] = None):
    """Runs a unit test case for the adder.

    Args:
      adder: The instance of `Adder` that is being tested.
      first: The first `dm_env.TimeStep` that is used to call
        `Adder.add_first()`.
      steps: A sequence of (action, timestep) tuples that are passed to
        `Adder.add()`.
      expected_items: The sequence of items that are expected to be created
        by calling the adder's `add_first()` method on `first` and `add()` on
        all of the elements in `steps`.
      signature: Signature that written items must be compatible with.
      pack_expected_items: Deprecated and not used. If true the expected items
        are given unpacked and need to be packed in a list before comparison.
      stack_sequence_fields: Whether to stack the sequence fields of the
        expected items before comparing to the observed items. Usually False
        for transition adders and True for both episode and sequence adders.
      repeat_episode_times: How many times to run an episode.
      end_behavior: How end of episode should be handled.
      item_transform: Transformation of item simulating the work done by the
        dataset pipeline on the learner in a real setup.
    """

    del pack_expected_items

    if not steps:
      raise ValueError('At least one step must be given.')

    has_extras = len(steps[0]) == 3
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

    # Force run the destructor to trigger the flushing of all pending items.
    getattr(adder, '__del__', lambda: None)()

    # Ending the episode should close the writer. No new writer should yet have
    # been created as it is constructed lazily.
    if end_behavior is not adders.EndBehavior.CONTINUE:
      self.assertEqual(self.num_episodes(), repeat_episode_times)

    # Make sure our expected and observed data match.
    observed_items = self.items()

    # Check matching number of items.
    self.assertEqual(len(expected_items), len(observed_items))

    # Check items are matching according to numpy's almost_equal.
    for expected_item, observed_item in zip(expected_items, observed_items):
      if stack_sequence_fields:
        expected_item = tree_utils.stack_sequence_fields(expected_item)

      # Apply the transformation which would be done by the dataset in a real
      # setup.
      if item_transform:
        observed_item = item_transform(observed_item)

      tree.map_structure(np.testing.assert_array_almost_equal,
                         tree.flatten(expected_item),
                         tree.flatten(observed_item))

    # Make sure the signature matches was is being written by Reverb.
    def _check_signature(spec: tf.TensorSpec, value: np.ndarray):
      self.assertTrue(spec.is_compatible_with(tf.convert_to_tensor(value)))

    # Check that it is possible to unpack observed using the signature.
    for item in observed_items:
      tree.map_structure(_check_signature, tree.flatten(signature),
                         tree.flatten(item))
