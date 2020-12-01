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

from typing import Iterator

from acme import adders as adders_lib
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
import dataclasses
import reverb


@dataclasses.dataclass
class ReverbReplay:
  server: reverb.Server
  adder: adders_lib.Adder
  data_iterator: Iterator[reverb.ReplaySample]
  client: reverb.Client


def make_reverb_prioritized_nstep_replay(
    environment_spec: specs.EnvironmentSpec,
    n_step: int = 1,
    batch_size: int = 32,
    max_replay_size: int = 100_000,
    min_replay_size: int = 1,
    priority_exponent: float = 0.,
    discount: float = 1.,
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
    prefetch_size: int = 4,
) -> ReverbReplay:
  """Creates a single-process replay infrastructure from an environment spec."""
  # Create a replay server to add data to. This uses no limiter behavior in
  # order to allow the Agent interface to handle it.
  replay_table = reverb.Table(
      name=replay_table_name,
      sampler=reverb.selectors.Prioritized(priority_exponent),
      remover=reverb.selectors.Fifo(),
      max_size=max_replay_size,
      rate_limiter=reverb.rate_limiters.MinSize(min_replay_size),
      signature=adders.NStepTransitionAdder.signature(
          environment_spec=environment_spec))
  server = reverb.Server([replay_table], port=None)

  # The adder is used to insert observations into replay.
  address = f'localhost:{server.port}'
  client = reverb.Client(address)
  adder = adders.NStepTransitionAdder(
      client=client, n_step=n_step, discount=discount)

  # The dataset provides an interface to sample from replay.
  data_iterator = datasets.make_reverb_dataset(
      table=replay_table_name,
      server_address=address,
      batch_size=batch_size,
      prefetch_size=prefetch_size,
      environment_spec=environment_spec,
      transition_adder=True,
  ).as_numpy_iterator()
  return ReverbReplay(server, adder, data_iterator, client)
