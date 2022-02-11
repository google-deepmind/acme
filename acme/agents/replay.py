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

"""Common tools for reverb replay."""

import dataclasses
from typing import Any, Callable, Dict, Iterator, Optional

from acme import adders as adders_lib
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as adders
import reverb


@dataclasses.dataclass
class ReverbReplay:
  server: reverb.Server
  adder: adders_lib.Adder
  data_iterator: Iterator[reverb.ReplaySample]
  client: Optional[reverb.Client] = None
  can_sample: Callable[[], bool] = lambda: True


def make_reverb_prioritized_nstep_replay(
    environment_spec: specs.EnvironmentSpec,
    extra_spec: types.NestedSpec = (),
    n_step: int = 1,
    batch_size: int = 32,
    max_replay_size: int = 100_000,
    min_replay_size: int = 1,
    discount: float = 1.,
    prefetch_size: int = 4,  # TODO(iosband): rationalize prefetch size.
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
    priority_exponent: Optional[float] = None,  # If None, default to uniform.
) -> ReverbReplay:
  """Creates a single-process replay infrastructure from an environment spec."""
  # Parsing priority exponent to determine uniform vs prioritized replay
  if priority_exponent is None:
    sampler = reverb.selectors.Uniform()
    priority_fns = {replay_table_name: lambda x: 1.}
  else:
    sampler = reverb.selectors.Prioritized(priority_exponent)
    priority_fns = None

  # Create a replay server to add data to. This uses no limiter behavior in
  # order to allow the Agent interface to handle it.
  replay_table = reverb.Table(
      name=replay_table_name,
      sampler=sampler,
      remover=reverb.selectors.Fifo(),
      max_size=max_replay_size,
      rate_limiter=reverb.rate_limiters.MinSize(min_replay_size),
      signature=adders.NStepTransitionAdder.signature(environment_spec,
                                                      extra_spec),
  )
  server = reverb.Server([replay_table], port=None)

  # The adder is used to insert observations into replay.
  address = f'localhost:{server.port}'
  client = reverb.Client(address)
  adder = adders.NStepTransitionAdder(
      client, n_step, discount, priority_fns=priority_fns)

  # The dataset provides an interface to sample from replay.
  data_iterator = datasets.make_reverb_dataset(
      table=replay_table_name,
      server_address=address,
      batch_size=batch_size,
      prefetch_size=prefetch_size,
  ).as_numpy_iterator()
  return ReverbReplay(server, adder, data_iterator, client=client)


def make_reverb_online_queue(
    environment_spec: specs.EnvironmentSpec,
    extra_spec: Dict[str, Any],
    max_queue_size: int,
    sequence_length: int,
    sequence_period: int,
    batch_size: int,
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
) -> ReverbReplay:
  """Creates a single process queue from an environment spec and extra_spec."""
  signature = adders.SequenceAdder.signature(environment_spec, extra_spec)
  queue = reverb.Table.queue(
      name=replay_table_name, max_size=max_queue_size, signature=signature)
  server = reverb.Server([queue], port=None)
  can_sample = lambda: queue.can_sample(batch_size)

  # Component to add things into replay.
  address = f'localhost:{server.port}'
  adder = adders.SequenceAdder(
      client=reverb.Client(address),
      period=sequence_period,
      sequence_length=sequence_length,
  )

  # The dataset object to learn from.
  # We don't use datasets.make_reverb_dataset() here to avoid interleaving
  # and prefetching, that doesn't work well with can_sample() check on update.
  dataset = reverb.TrajectoryDataset.from_table_signature(
      server_address=address,
      table=replay_table_name,
      max_in_flight_samples_per_worker=1,
  )
  dataset = dataset.batch(batch_size, drop_remainder=True)
  data_iterator = dataset.as_numpy_iterator()
  return ReverbReplay(server, adder, data_iterator, can_sample=can_sample)


def make_reverb_prioritized_sequence_replay(
    environment_spec: specs.EnvironmentSpec,
    extra_spec: types.NestedSpec = (),
    batch_size: int = 32,
    max_replay_size: int = 100_000,
    min_replay_size: int = 1,
    priority_exponent: float = 0.,
    burn_in_length: int = 40,
    sequence_length: int = 80,
    sequence_period: int = 40,
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
    prefetch_size: int = 4,
) -> ReverbReplay:
  """Single-process replay for sequence data from an environment spec."""
  # Create a replay server to add data to. This uses no limiter behavior in
  # order to allow the Agent interface to handle it.
  replay_table = reverb.Table(
      name=replay_table_name,
      sampler=reverb.selectors.Prioritized(priority_exponent),
      remover=reverb.selectors.Fifo(),
      max_size=max_replay_size,
      rate_limiter=reverb.rate_limiters.MinSize(min_replay_size),
      signature=adders.SequenceAdder.signature(environment_spec, extra_spec),
  )
  server = reverb.Server([replay_table], port=None)

  # The adder is used to insert observations into replay.
  address = f'localhost:{server.port}'
  client = reverb.Client(address)
  sequence_length = burn_in_length + sequence_length + 1
  adder = adders.SequenceAdder(
      client=client,
      period=sequence_period,
      sequence_length=sequence_length,
      delta_encoded=True,
  )

  # The dataset provides an interface to sample from replay.
  data_iterator = datasets.make_reverb_dataset(
      table=replay_table_name,
      server_address=address,
      batch_size=batch_size,
      prefetch_size=prefetch_size,
  ).as_numpy_iterator()
  return ReverbReplay(server, adder, data_iterator, client)
