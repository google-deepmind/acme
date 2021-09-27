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

from typing import Any, Optional

from acme import specs
from acme.agents import agent
from acme.agents.jax import builders
from acme.jax import utils
from acme.tf import savers
from acme.utils import counting
import jax
import reverb


class LocalLayout(agent.Agent):
  """An Agent that runs an algorithm defined by 'builder' on a single machine.
  """

  def __init__(
      self,
      seed: int,
      environment_spec: specs.EnvironmentSpec,
      builder: builders.GenericActorLearnerBuilder,
      networks: Any,
      policy_network: Any,
      workdir: Optional[str] = '~/acme',
      min_replay_size: int = 1000,
      samples_per_insert: float = 256.0,
      batch_size: int = 256,
      num_sgd_steps_per_step: int = 1,
      prefetch_size: int = 1,
      device_prefetch: bool = True,
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
      workdir: if provided saves the state of the learner and the counter
        (if the counter is not None) into workdir.
      min_replay_size: minimum replay size before updating.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      batch_size: batch size for updates.
      num_sgd_steps_per_step: how many sgd steps a learner does per 'step' call.
        For performance reasons (especially to reduce TPU host-device transfer
        times) it is performance-beneficial to do multiple sgd updates at once,
        provided that it does not hurt the training, which needs to be verified
        empirically for each environment.
      prefetch_size: whether to prefetch iterator.
      device_prefetch: whether prefetching should happen to a device.
      counter: counter object used to keep track of steps.
      checkpoint: boolean indicating whether to checkpoint the learner
        and the counter (if the counter is not None).
    """
    if prefetch_size < 0:
      raise ValueError(f'Prefetch size={prefetch_size} should be non negative')

    key = jax.random.PRNGKey(seed)

    # Create the replay server and grab its address.
    replay_tables = builder.make_replay_tables(environment_spec)
    replay_server = reverb.Server(replay_tables, port=None)
    replay_client = reverb.Client(f'localhost:{replay_server.port}')

    # Create actor, dataset, and learner for generating, storing, and consuming
    # data respectively.
    adder = builder.make_adder(replay_client)

    def _is_reverb_queue(reverb_table: reverb.Table,
                         reverb_client: reverb.Client) -> bool:
      """Returns True iff the Reverb Table is actually a queue."""
      # TODO(sinopalnikov): make it more generic and check for a table that
      # needs special handling on update.
      info = reverb_client.server_info()
      table_info = info[reverb_table.name]
      is_queue = (
          table_info.max_times_sampled == 1 and
          table_info.sampler_options.fifo and
          table_info.remover_options.fifo)
      return is_queue

    is_reverb_queue = any(_is_reverb_queue(table, replay_client)
                          for table in replay_tables)

    dataset = builder.make_dataset_iterator(replay_client)
    if prefetch_size > 1:
      device = jax.devices()[0] if device_prefetch else None
      dataset = utils.prefetch(dataset, buffer_size=prefetch_size,
                               device=device)
    learner_key, key = jax.random.split(key)
    learner = builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=dataset,
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
        actor_key, policy_network, adder, variable_source=learner)
    self._custom_update_fn = None
    if is_reverb_queue:
      # Reverb queue requires special handling on update: custom logic to
      # decide when it is safe to make a learner step. This is only needed for
      # the local agent, where the actor and the learner are running
      # synchronously and the learner will deadlock if it makes a step with
      # no data available.
      def custom_update():
        should_update_actor = False
        # Run a number of learner steps (usually gradient steps).
        # TODO(raveman): This is wrong. When running multi-level learners,
        # different levels might have different batch sizes. Find a solution.
        while all(table.can_sample(batch_size) for table in replay_tables):
          learner.step()
          should_update_actor = True

        if should_update_actor:
          # "wait=True" to make it more onpolicy
          actor.update(wait=True)

      self._custom_update_fn = custom_update

    effective_batch_size = batch_size * num_sgd_steps_per_step
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(effective_batch_size, min_replay_size),
        observations_per_step=float(effective_batch_size) / samples_per_insert)

    # Save the replay so we don't garbage collect it.
    self._replay_server = replay_server

  def update(self):
    if self._custom_update_fn:
      self._custom_update_fn()
    else:
      super().update()
    if self._checkpointer:
      self._checkpointer.save()

  def save(self):
    """Checkpoint the state of the agent."""
    if self._checkpointer:
      self._checkpointer.save(force=True)
