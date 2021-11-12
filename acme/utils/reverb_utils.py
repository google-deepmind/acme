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

"""Reverb utils.

Contains functions manipulating reverb tables and samples.
"""

from acme import types
import jax
import numpy as np
import reverb
from reverb import item_selectors
from reverb import rate_limiters
from reverb import reverb_types
import tree


def make_replay_table_from_info(
    table_info: reverb_types.TableInfo) -> reverb.Table:
  """Build a replay table out of its specs in a TableInfo.

  Args:
    table_info: A TableInfo containing the Table specs.

  Returns:
    A reverb replay table matching the info specs.
  """
  sampler = _make_selector_from_key_distribution_options(
      table_info.sampler_options)
  remover = _make_selector_from_key_distribution_options(
      table_info.remover_options)
  rate_limiter = _make_rate_limiter_from_rate_limiter_info(
      table_info.rate_limiter_info)
  return reverb.Table(
      name=table_info.name,
      sampler=sampler,
      remover=remover,
      max_size=table_info.max_size,
      rate_limiter=rate_limiter,
      max_times_sampled=table_info.max_times_sampled,
      signature=table_info.signature)


def _make_selector_from_key_distribution_options(
    options) -> reverb_types.SelectorType:
  """Returns a Selector from its KeyDistributionOptions description."""
  one_of = options.WhichOneof('distribution')
  if one_of == 'fifo':
    return item_selectors.Fifo()
  if one_of == 'uniform':
    return item_selectors.Uniform()
  if one_of == 'prioritized':
    return item_selectors.Prioritized(options.prioritized.priority_exponent)
  if one_of == 'heap':
    if options.heap.min_heap:
      return item_selectors.MinHeap()
    return item_selectors.MaxHeap()
  if one_of == 'lifo':
    return item_selectors.Lifo()
  raise ValueError(f'Unknown distribution field: {one_of}')


def _make_rate_limiter_from_rate_limiter_info(
    info) -> rate_limiters.RateLimiter:
  return rate_limiters.SampleToInsertRatio(
      samples_per_insert=info.samples_per_insert,
      min_size_to_sample=info.min_size_to_sample,
      error_buffer=(info.min_diff, info.max_diff))


def replay_sample_to_sars_transition(
    sample: reverb.ReplaySample,
    is_sequence: bool,
    strip_last_transition: bool = False,
    flatten_batch: bool = False) -> types.Transition:
  """Converts the replay sample to a types.Transition.

  NB: If is_sequence is True then the last next_observation of each sequence is
  rubbish. Don't train on it.

  Args:
    sample: The replay sample
    is_sequence: If False we expect the sample data to match the
      types.Transition already. Otherwise we expect a batch of sequences of
      steps.
    strip_last_transition: If True and is_sequence, the last transition will be
      stripped as its next_observation field is incorrect.
    flatten_batch: If True and is_sequence, the two batch dimensions will be
      flatten to one.

  Returns:
    A types.Transition built from the sample data.
    If is_sequence and strip_last_transition are both True, the output will be
    smaller than the output as the last transition of every sequence will have
    been removed.
  """
  if not is_sequence:
    return types.Transition(*sample.data)
  # Note that the last next_observation is invalid.
  steps = sample.data
  def roll(observation):
    return np.roll(observation, shift=-1, axis=1)
  transitions = types.Transition(
      observation=steps.observation,
      action=steps.action,
      reward=steps.reward,
      discount=steps.discount,
      next_observation=tree.map_structure(roll, steps.observation))
  if strip_last_transition:
    # We remove the last transition as its next_observation field is incorrect.
    # It has been obtained by rolling the observation field, such that
    # transitions.next_observations[:, -1] is transitions.observations[:, 0]
    transitions = jax.tree_map(lambda x: x[:, :-1, ...], transitions)
  if flatten_batch:
    # Merge the 2 leading batch dimensions into 1.
    transitions = jax.tree_map(lambda x: np.reshape(x, (-1,) + x.shape[2:]),
                               transitions)
  return transitions
