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

import sys
from typing import Any, Optional

from acme import specs
from acme.agents import agent
from acme.agents.jax import builders
from acme.jax import utils
from acme.tf import savers
from acme.utils import counting
from acme.utils import loggers
import jax
import reverb


class LocalLayout(agent.Agent):
  """An Agent that runs an algorithm defined by 'builder' on a single machine.
  """

  def __init__(
      self,
      seed: int,
      environment_spec: specs.EnvironmentSpec,
      builder: builders.ActorLearnerBuilder,
      networks: Any,
      policy_network: Any,
      learner_logger: Optional[loggers.Logger] = None,
      workdir: Optional[str] = '~/acme',
      batch_size: int = 256,
      num_sgd_steps_per_step: int = 1,
      prefetch_size: int = 1,
      counter: Optional[counting.Counter] = None,
      checkpoint: bool = True,
  ):
    """Initialize the agent.

    Args:
      seed: A random seed to use for this layout instance.
      environment_spec: description of the actions, observations, etc.
      builder: builder defining an RL algorithm to train.
      networks: network objects to be passed to the learner.
      policy_network: function that given an observation returns actions.
      learner_logger: logger used by the learner.
      workdir: if provided saves the state of the learner and the counter
        (if the counter is not None) into workdir.
      batch_size: batch size for updates.
      num_sgd_steps_per_step: how many sgd steps a learner does per 'step' call.
        For performance reasons (especially to reduce TPU host-device transfer
        times) it is performance-beneficial to do multiple sgd updates at once,
        provided that it does not hurt the training, which needs to be verified
        empirically for each environment.
      prefetch_size: whether to prefetch iterator.
      counter: counter object used to keep track of steps.
      checkpoint: boolean indicating whether to checkpoint the learner
        and the counter (if the counter is not None).
    """
    if prefetch_size < 0:
      raise ValueError(f'Prefetch size={prefetch_size} should be non negative')

    key = jax.random.PRNGKey(seed)

    # Create the replay server and grab its address.
    replay_tables = builder.make_replay_tables(environment_spec, policy_network)

    # Disable blocking of inserts by tables' rate limiters, as LocalLayout
    # agents run inserts and sampling from the same thread and blocked insert
    # would result in a hang.
    new_tables = []
    for table in replay_tables:
      rl_info = table.info.rate_limiter_info
      rate_limiter = reverb.rate_limiters.RateLimiter(
          samples_per_insert=rl_info.samples_per_insert,
          min_size_to_sample=rl_info.min_size_to_sample,
          min_diff=rl_info.min_diff,
          max_diff=sys.float_info.max)
      new_tables.append(table.replace(rate_limiter=rate_limiter))
    replay_tables = new_tables

    replay_server = reverb.Server(replay_tables, port=None)
    replay_client = reverb.Client(f'localhost:{replay_server.port}')

    # Create actor, dataset, and learner for generating, storing, and consuming
    # data respectively.
    adder = builder.make_adder(replay_client)

    dataset = builder.make_dataset_iterator(replay_client)
    # We always use prefetch, as it provides an iterator with additional
    # 'ready' method.
    dataset = utils.prefetch(dataset, buffer_size=prefetch_size)
    learner_key, key = jax.random.split(key)
    learner = builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=dataset,
        logger_fn=(
            lambda label, steps_key=None, task_instance=None: learner_logger),
        environment_spec=environment_spec,
        replay_client=replay_client,
        counter=counter)
    if not checkpoint or workdir is None:
      self._checkpointer = None
    else:
      objects_to_save = {'learner': learner}
      if counter is not None:
        objects_to_save.update({'counter': counter})
      self._checkpointer = savers.Checkpointer(
          objects_to_save,
          time_delta_minutes=30,
          subdirectory='learner',
          directory=workdir,
          add_uid=(workdir == '~/acme'))

    actor_key, key = jax.random.split(key)
    actor = builder.make_actor(
        actor_key,
        policy_network,
        environment_spec,
        variable_source=learner,
        adder=adder)

    super().__init__(
        actor=actor,
        learner=learner,
        iterator=dataset,
        replay_tables=replay_tables)

    # Save the replay so we don't garbage collect it.
    self._replay_server = replay_server

  def update(self):
    super().update()
    if self._checkpointer:
      self._checkpointer.save()

  def save(self):
    """Checkpoint the state of the agent."""
    if self._checkpointer:
      self._checkpointer.save(force=True)
