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

"""Builder enabling off-policy algorithms to learn from demonstrations."""

from typing import Any, Callable, Generic, Iterator, Tuple

from acme.agents.jax import builders
from acme.agents.jax.lfd import config as lfd_config
from acme.agents.jax.lfd import lfd_adder
import dm_env


LfdStep = Tuple[Any, dm_env.TimeStep]


class LfdBuilder(builders.ActorLearnerBuilder[builders.Networks,
                                              builders.Policy,
                                              builders.Sample,],
                 Generic[builders.Networks, builders.Policy, builders.Sample]):
  """Builder that enables Learning From demonstrations.

  This builder is not self contained and requires an underlying builder
  implementing an off-policy algorithm.
  """

  def __init__(self, builder: builders.ActorLearnerBuilder[builders.Networks,
                                                           builders.Policy,
                                                           builders.Sample],
               demonstrations_factory: Callable[[], Iterator[LfdStep]],
               config: lfd_config.LfdConfig):
    """LfdBuilder constructor.

    Args:
      builder: The underlying builder implementing the off-policy algorithm.
      demonstrations_factory: Factory returning an infinite stream (as an
        iterator) of (action, next_timesteps). Episode boundaries in this stream
        are given by timestep.first() and timestep.last(). Note that in the
        distributed version of this algorithm, each actor is mixing the same
        demonstrations with its online experience. This effectively results in
        the demonstrations being replicated in the replay buffer as many times
        as the number of actors being used.
      config: LfD configuration.
    """
    self._builder = builder
    self._demonstrations_factory = demonstrations_factory
    self._config = config

  def make_replay_tables(self, *args, **kwargs):
    return self._builder.make_replay_tables(*args, **kwargs)

  def make_dataset_iterator(self, *args, **kwargs):
    return self._builder.make_dataset_iterator(*args, **kwargs)

  def make_adder(self, *args, **kwargs):
    demonstrations = self._demonstrations_factory()
    return lfd_adder.LfdAdder(self._builder.make_adder(*args, **kwargs),
                              demonstrations,
                              self._config.initial_insert_count,
                              self._config.demonstration_ratio)

  def make_actor(self, *args, **kwargs):
    return self._builder.make_actor(*args, **kwargs)

  def make_learner(self, *args, **kwargs):
    return self._builder.make_learner(*args, **kwargs)

  def make_policy(self, *args, **kwargs):
    return self._builder.make_policy(*args, **kwargs)
