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

"""Local agent based on builders."""

from typing import Any

from acme import specs
from acme.agents import agent
from acme.agents import builders
from acme.agents.jax import actors
from acme.utils import counting
from acme.utils import loggers
import reverb


class LocalLayout(agent.Agent):
  """An Agent that runs an algorithm defined by 'builder' on a single machine.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      builder: builders.ActorLearnerBuilder,
      networks: Any,
      policy_network: actors.FeedForwardPolicy,
      min_replay_size: int = 1000,
      samples_per_insert: float = 256.0,
      batch_size: int = 256,
      num_sgd_steps_per_step: int = 1,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
    ):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      builder: builder defining an RL algorithm to train.
      networks: network objects to be passed to the learner.
      policy_network: function that given an observation returns actions.
      min_replay_size: minimum replay size before updating.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      batch_size: batch size for updates.
      num_sgd_steps_per_step: how many sgd steps a learner does per 'step' call.
        For performance reasons (especially to reduce TPU host-device transfer
        times) it is performance-beneficial to do multiple sgd updates at once,
        provided that it does not hurt the training, which needs to be verified
        empirically for each environment.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
    """
    # Create the replay server and grab its address.
    replay_tables = builder.make_replay_tables(environment_spec)
    replay_server = reverb.Server(replay_tables, port=None)
    replay_client = reverb.Client(f'localhost:{replay_server.port}')

    # Create actor, dataset, and learner for generating, storing, and consuming
    # data respectively.
    adder = builder.make_adder(replay_client)
    dataset = builder.make_dataset_iterator(replay_client)
    learner = builder.make_learner(networks=networks, dataset=dataset,
                                   replay_client=replay_client, counter=counter,
                                   logger=logger, checkpoint=checkpoint)
    actor = builder.make_actor(policy_network, adder, variable_source=learner)

    effective_batch_size = batch_size * num_sgd_steps_per_step
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(effective_batch_size, min_replay_size),
        observations_per_step=float(effective_batch_size) / samples_per_insert)

    # Save the replay so we don't garbage collect it.
    self._replay_server = replay_server
