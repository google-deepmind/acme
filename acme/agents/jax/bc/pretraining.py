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

"""Tools to train a policy network with BC."""
from typing import Callable, Iterator

from acme import types
from acme.agents.jax.bc import learning
from acme.agents.jax.bc import losses
from acme.agents.jax.bc import networks as bc_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
import jax
import optax


def train_with_bc(make_demonstrations: Callable[[int],
                                                Iterator[types.Transition]],
                  networks: bc_networks.BCNetworks,
                  loss: losses.BCLoss,
                  num_steps: int = 100000) -> networks_lib.Params:
  """Trains the given network with BC and returns the params.

  Args:
    make_demonstrations: A function (batch_size) -> iterator with demonstrations
      to be imitated.
    networks: Network taking (params, obs, is_training, key) as input
    loss: BC loss to use.
    num_steps: number of training steps

  Returns:
    The trained network params.
  """
  demonstration_iterator = make_demonstrations(256)
  prefetching_iterator = utils.sharded_prefetch(
      demonstration_iterator,
      buffer_size=2,
      num_threads=jax.local_device_count())

  learner = learning.BCLearner(
      networks=networks,
      random_key=jax.random.PRNGKey(0),
      loss_fn=loss,
      prefetching_iterator=prefetching_iterator,
      optimizer=optax.adam(1e-4),
      num_sgd_steps_per_step=1)

  # Train the agent
  for _ in range(num_steps):
    learner.step()

  return learner.get_variables(['policy'])[0]
