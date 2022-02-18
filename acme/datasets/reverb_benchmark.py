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

"""Reverb dataset benchmark.

Note: this a no-GRPC layer setup.
"""

import time
from typing import Sequence

from absl import app
from absl import logging
from acme import adders
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.datasets import reverb as datasets
from acme.testing import fakes
import numpy as np
import reverb
from reverb import rate_limiters


def make_replay_tables(environment_spec: specs.EnvironmentSpec
                      ) -> Sequence[reverb.Table]:
  """Create tables to insert data into."""
  return [
      reverb.Table(
          name='default',
          sampler=reverb.selectors.Uniform(),
          remover=reverb.selectors.Fifo(),
          max_size=1000000,
          rate_limiter=rate_limiters.MinSize(1),
          signature=adders_reverb.NStepTransitionAdder.signature(
              environment_spec))
  ]


def make_adder(replay_client: reverb.Client) -> adders.Adder:
  return adders_reverb.NStepTransitionAdder(
      priority_fns={'default': None},
      client=replay_client,
      n_step=1,
      discount=1)


def main(_):
  environment = fakes.ContinuousEnvironment(action_dim=8,
                                            observation_dim=87,
                                            episode_length=10000000)
  spec = specs.make_environment_spec(environment)
  replay_tables = make_replay_tables(spec)
  replay_server = reverb.Server(replay_tables, port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')
  adder = make_adder(replay_client)

  timestep = environment.reset()
  adder.add_first(timestep)
  # TODO(raveman): Consider also filling the table to say 1M (too slow).
  for steps in range(10000):
    if steps % 1000 == 0:
      logging.info('Processed %s steps', steps)
    action = np.asarray(np.random.uniform(-1, 1, (8,)), dtype=np.float32)
    next_timestep = environment.step(action)
    adder.add(action, next_timestep, extras=())

  for batch_size in [256, 256 * 8, 256 * 64]:
    for prefetch_size in [0, 1, 4]:
      print(f'Processing batch_size={batch_size} prefetch_size={prefetch_size}')
      ds = datasets.make_reverb_dataset(
          table='default',
          server_address=replay_client.server_address,
          batch_size=batch_size,
          prefetch_size=prefetch_size,
      )
      it = ds.as_numpy_iterator()

      for iteration in range(3):
        t = time.time()
        for _ in range(1000):
          _ = next(it)
        print(f'Iteration {iteration} finished in {time.time() - t}s')


if __name__ == '__main__':
  app.run(main)
