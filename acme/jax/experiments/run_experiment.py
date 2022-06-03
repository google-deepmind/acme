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

import math
import sys
from typing import List

import acme
from acme import core
from acme import specs
from acme import types
from acme.jax import utils
from acme.jax.experiments import config
from acme.utils import counting
import dm_env
import jax
import reverb


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
               replay_tables: List[reverb.Table]):
    self._actor = actor
    self._learner = learner
    self._iterator = iterator
    self._replay_tables = replay_tables
    self._batch_size_upper_bounds = [1_000_000_000] * len(replay_tables)

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    return self._actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    self._actor.observe(action, next_timestep)

  def _has_data_for_training(self):
    if self._iterator.ready():
      return True
    for (table, batch_size) in zip(self._replay_tables,
                                   self._batch_size_upper_bounds):
      if not table.can_sample(batch_size):
        return False
    return True

  def update(self):
    # Perform learner steps as long as iterator has data.
    update_actor = False
    while self._has_data_for_training():
      # Run learner steps (usually means gradient steps).
      total_batches = self._iterator.retrieved_elements()
      self._learner.step()
      current_batches = self._iterator.retrieved_elements() - total_batches
      assert current_batches == 1, (
          'Learner step must retrieve exactly one element from the iterator'
          f' (retrieved {current_batches}). Otherwise agent can deadlock.')
      self._batch_size_upper_bounds = [
          math.ceil(t.info.rate_limiter_info.sample_stats.completed /
                    (total_batches + 1)) for t in self._replay_tables
      ]
      update_actor = True
    if update_actor:
      # Update the actor weights only when learner was updated.
      self._actor.update()


def _disable_insert_blocking(table: reverb.Table):
  rate_limiter_info = table.info.rate_limiter_info
  rate_limiter = reverb.rate_limiters.RateLimiter(
      samples_per_insert=rate_limiter_info.samples_per_insert,
      min_size_to_sample=rate_limiter_info.min_size_to_sample,
      min_diff=rate_limiter_info.min_diff,
      max_diff=sys.float_info.max)
  return table.replace(rate_limiter=rate_limiter)


def run_experiment(experiment: config.Config,
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
  policy = experiment.policy_network_factory(networks)

  # Create the replay server and grab its address.
  replay_tables = experiment.builder.make_replay_tables(environment_spec,
                                                        policy)

  # Disable blocking of inserts by tables' rate limiters, as this function
  # executes learning (sampling from the table) and data generation
  # (inserting into the table) sequentially from the same thread
  # which could result in blocked insert making the algorithm hang.
  replay_tables = [_disable_insert_blocking(table) for table in replay_tables]

  replay_server = reverb.Server(replay_tables, port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')

  # Create actor, dataset, and learner for generating, storing, and consuming
  # data respectively.
  adder = experiment.builder.make_adder(replay_client)

  # Parent counter allows to share step counts between train and eval loops and
  # the learner, so that it is possible to plot for example evaluator's return
  # value as a function of the number of training episodes.
  parent_counter = counting.Counter(time_delta=0.)

  dataset = experiment.builder.make_dataset_iterator(replay_client)
  # We always use prefetch, as it provides an iterator with additional
  # 'ready' method.
  dataset = utils.prefetch(dataset, buffer_size=1)
  learner_key, key = jax.random.split(key)
  learner_logger = experiment.logger_factory('learner', 'learner_steps', 0)
  learner = experiment.builder.make_learner(
      random_key=learner_key,
      networks=networks,
      dataset=dataset,
      logger=learner_logger,
      environment_spec=environment_spec,
      replay_client=replay_client,
      counter=counting.Counter(parent_counter, prefix='learner', time_delta=0.))

  actor_key, key = jax.random.split(key)
  actor = experiment.builder.make_actor(
      actor_key, policy, environment_spec, variable_source=learner, adder=adder)

  # Create the environment loop used for training.
  train_logger = experiment.logger_factory('train', 'train_steps', 0)

  # Replace the actor with a LearningActor. This makes sure that every time
  # that `update` is called on the actor it checks to see whether there is
  # any new data to learn from and if so it runs a learner step. The rate
  # at which new data is released is controlled by the replay table's
  # rate_limiter which is created by the builder.make_replay_tables call above.
  actor = _LearningActor(actor, learner, dataset, replay_tables)

  train_loop = acme.EnvironmentLoop(
      environment,
      actor,
      counter=counting.Counter(parent_counter, prefix='train', time_delta=0.),
      logger=train_logger)

  eval_loop = None
  if experiment.eval_policy_network_factory:
    # Create the evaluation actor and loop.
    eval_logger = experiment.logger_factory('eval', 'eval_steps', 0)
    eval_actor = experiment.builder.make_actor(
        random_key=jax.random.PRNGKey(experiment.seed),
        policy=experiment.eval_policy_network_factory(networks),
        environment_spec=environment_spec,
        variable_source=learner)
    eval_loop = acme.EnvironmentLoop(
        environment,
        eval_actor,
        counter=counting.Counter(parent_counter, prefix='eval', time_delta=0.),
        logger=eval_logger)

  steps = 0
  while steps < experiment.max_number_of_steps:
    if eval_loop:
      eval_loop.run(num_episodes=num_eval_episodes)
    train_loop.run(num_steps=eval_every)
    steps += eval_every
  if eval_loop:
    eval_loop.run(num_episodes=num_eval_episodes)
