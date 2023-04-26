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

"""Runners used for executing local agents."""

import sys
import time
from typing import Optional, Sequence, Tuple

import acme
from acme import core
from acme import specs
from acme import types
from acme.jax import utils
from acme.jax.experiments import config
from acme.tf import savers
from acme.utils import counting
import dm_env
import jax
import reverb


def run_experiment(experiment: config.ExperimentConfig,
                   eval_every: int = 100,
                   num_eval_episodes: int = 1):
  """Runs a simple, single-threaded training loop using the default evaluators.

  It targets simplicity of the code and so only the basic features of the
  ExperimentConfig are supported.

  Arguments:
    experiment: Definition and configuration of the agent to run.
    eval_every: After how many actor steps to perform evaluation.
    num_eval_episodes: How many evaluation episodes to execute at each
      evaluation step.
  """

  key = jax.random.PRNGKey(experiment.seed)

  # Create the environment and get its spec.
  environment = experiment.environment_factory(experiment.seed)
  environment_spec = experiment.environment_spec or specs.make_environment_spec(
      environment)

  # Create the networks and policy.
  networks = experiment.network_factory(environment_spec)
  policy = config.make_policy(
      experiment=experiment,
      networks=networks,
      environment_spec=environment_spec,
      evaluation=False)

  # Create the replay server and grab its address.
  replay_tables = experiment.builder.make_replay_tables(environment_spec,
                                                        policy)

  # Disable blocking of inserts by tables' rate limiters, as this function
  # executes learning (sampling from the table) and data generation
  # (inserting into the table) sequentially from the same thread
  # which could result in blocked insert making the algorithm hang.
  replay_tables, rate_limiters_max_diff = _disable_insert_blocking(
      replay_tables)

  replay_server = reverb.Server(replay_tables, port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')

  # Parent counter allows to share step counts between train and eval loops and
  # the learner, so that it is possible to plot for example evaluator's return
  # value as a function of the number of training episodes.
  parent_counter = counting.Counter(time_delta=0.)

  dataset = experiment.builder.make_dataset_iterator(replay_client)
  # We always use prefetch as it provides an iterator with an additional
  # 'ready' method.
  dataset = utils.prefetch(dataset, buffer_size=1)

  # Create actor, adder, and learner for generating, storing, and consuming
  # data respectively.
  # NOTE: These are created in reverse order as the actor needs to be given the
  # adder and the learner (as a source of variables).
  learner_key, key = jax.random.split(key)
  learner = experiment.builder.make_learner(
      random_key=learner_key,
      networks=networks,
      dataset=dataset,
      logger_fn=experiment.logger_factory,
      environment_spec=environment_spec,
      replay_client=replay_client,
      counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.))

  adder = experiment.builder.make_adder(replay_client, environment_spec, policy)

  actor_key, key = jax.random.split(key)
  actor = experiment.builder.make_actor(
      actor_key, policy, environment_spec, variable_source=learner, adder=adder)

  # Create the environment loop used for training.
  train_counter = counting.Counter(
      parent_counter, prefix='actor', time_delta=0.)
  train_logger = experiment.logger_factory('actor',
                                           train_counter.get_steps_key(), 0)

  checkpointer = None
  if experiment.checkpointing is not None:
    checkpointing = experiment.checkpointing
    checkpointer = savers.Checkpointer(
        objects_to_save={'learner': learner, 'counter': parent_counter},
        time_delta_minutes=checkpointing.time_delta_minutes,
        directory=checkpointing.directory,
        add_uid=checkpointing.add_uid,
        max_to_keep=checkpointing.max_to_keep,
        keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
        checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
    )

  # Replace the actor with a LearningActor. This makes sure that every time
  # that `update` is called on the actor it checks to see whether there is
  # any new data to learn from and if so it runs a learner step. The rate
  # at which new data is released is controlled by the replay table's
  # rate_limiter which is created by the builder.make_replay_tables call above.
  actor = _LearningActor(actor, learner, dataset, replay_tables,
                         rate_limiters_max_diff, checkpointer)

  train_loop = acme.EnvironmentLoop(
      environment,
      actor,
      counter=train_counter,
      logger=train_logger,
      observers=experiment.observers)

  max_num_actor_steps = (
      experiment.max_num_actor_steps -
      parent_counter.get_counts().get(train_counter.get_steps_key(), 0))

  if num_eval_episodes == 0:
    # No evaluation. Just run the training loop.
    train_loop.run(num_steps=max_num_actor_steps)
    return

  # Create the evaluation actor and loop.
  eval_counter = counting.Counter(
      parent_counter, prefix='evaluator', time_delta=0.)
  eval_logger = experiment.logger_factory('evaluator',
                                          eval_counter.get_steps_key(), 0)
  eval_policy = config.make_policy(
      experiment=experiment,
      networks=networks,
      environment_spec=environment_spec,
      evaluation=True)
  eval_actor = experiment.builder.make_actor(
      random_key=jax.random.PRNGKey(experiment.seed),
      policy=eval_policy,
      environment_spec=environment_spec,
      variable_source=learner)
  eval_loop = acme.EnvironmentLoop(
      environment,
      eval_actor,
      counter=eval_counter,
      logger=eval_logger,
      observers=experiment.observers)

  steps = 0
  while steps < max_num_actor_steps:
    eval_loop.run(num_episodes=num_eval_episodes)
    num_steps = min(eval_every, max_num_actor_steps - steps)
    steps += train_loop.run(num_steps=num_steps)
  eval_loop.run(num_episodes=num_eval_episodes)

  environment.close()


class _LearningActor(core.Actor):
  """Actor which learns (updates its parameters) when `update` is called.

  This combines a base actor and a learner. Whenever `update` is called
  on the wrapping actor the learner will take a step (e.g. one step of gradient
  descent) as long as there is data available for training
  (provided iterator and replay_tables are used to check for that).
  Selecting actions and making observations are handled by the base actor.
  Intended to be used by the `run_experiment` only.
  """

  def __init__(self, actor: core.Actor, learner: core.Learner,
               iterator: core.PrefetchingIterator,
               replay_tables: Sequence[reverb.Table],
               sample_sizes: Sequence[int],
               checkpointer: Optional[savers.Checkpointer]):
    """Initializes _LearningActor.

    Args:
      actor: Actor to be wrapped.
      learner: Learner on which step() is to be called when there is data.
      iterator: Iterator used by the Learner to fetch training data.
      replay_tables: Collection of tables from which Learner fetches data
        through the iterator.
      sample_sizes: For each table from `replay_tables`, how many elements the
        table should have available for sampling to wait for the `iterator` to
        prefetch a batch of data. Otherwise more experience needs to be
        collected by the actor.
      checkpointer: Checkpointer to save the state on update.
    """
    self._actor = actor
    self._learner = learner
    self._iterator = iterator
    self._replay_tables = replay_tables
    self._sample_sizes = sample_sizes
    self._learner_steps = 0
    self._checkpointer = checkpointer

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    return self._actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    self._actor.observe(action, next_timestep)

  def _maybe_train(self):
    trained = False
    while True:
      if self._iterator.ready():
        self._learner.step()
        batches = self._iterator.retrieved_elements() - self._learner_steps
        self._learner_steps += 1
        assert batches == 1, (
            'Learner step must retrieve exactly one element from the iterator'
            f' (retrieved {batches}). Otherwise agent can deadlock. Example '
            'cause is that your chosen agent'
            's Builder has a `make_learner` '
            'factory that prefetches the data but it shouldn'
            't.')
        trained = True
      else:
        # Wait for the iterator to fetch more data from the table(s) only
        # if there plenty of data to sample from each table.
        for table, sample_size in zip(self._replay_tables, self._sample_sizes):
          if not table.can_sample(sample_size):
            return trained
        # Let iterator's prefetching thread get data from the table(s).
        time.sleep(0.001)

  def update(self):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    if self._maybe_train():
      # Update the actor weights only when learner was updated.
      self._actor.update()
    if self._checkpointer:
      self._checkpointer.save()


def _disable_insert_blocking(
    tables: Sequence[reverb.Table]
) -> Tuple[Sequence[reverb.Table], Sequence[int]]:
  """Disables blocking of insert operations for a given collection of tables."""
  modified_tables = []
  sample_sizes = []
  for table in tables:
    rate_limiter_info = table.info.rate_limiter_info
    rate_limiter = reverb.rate_limiters.RateLimiter(
        samples_per_insert=rate_limiter_info.samples_per_insert,
        min_size_to_sample=rate_limiter_info.min_size_to_sample,
        min_diff=rate_limiter_info.min_diff,
        max_diff=sys.float_info.max)
    modified_tables.append(table.replace(rate_limiter=rate_limiter))
    # Target the middle of the rate limiter's insert-sample balance window.
    sample_sizes.append(
        max(1, int(
            (rate_limiter_info.max_diff - rate_limiter_info.min_diff) / 2)))
  return modified_tables, sample_sizes
