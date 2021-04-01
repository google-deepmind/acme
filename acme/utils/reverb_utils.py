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
import reverb
from reverb import item_selectors
from reverb import rate_limiters
from reverb import reverb_types


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
