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
from typing import Sequence, Tuple

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
  replay_tables, rate_limiters_max_diff = _disable_insert_blocking(
      replay_tables)

  replay_server = reverb.Server(replay_tables, port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')

  # Parent counter allows to share step counts between train and eval loops and
  # the learner, so that it is possible to plot for example evaluator's return
  # value as a function of the number of training episodes.
  parent_counter = counting.Counter(time_delta=0.)

  # Create actor, and learner for generating, storing, and consuming
  # data respectively.
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

  adder = experiment.builder.make_adder(replay_client)
  adder = _TrainingAdder(adder, learner, dataset, replay_tables,
                         rate_limiters_max_diff)

  actor_key, key = jax.random.split(key)
  actor = experiment.builder.make_actor(
      actor_key, policy, environment_spec, variable_source=learner, adder=adder)

  # Create the environment loop used for training.
  train_logger = experiment.logger_factory('train', 'train_steps', 0)

  train_loop = acme.EnvironmentLoop(
      environment,
      actor,
      counter=counting.Counter(parent_counter, prefix='train', time_delta=0.),
      logger=train_logger,
      observers=experiment.observers)

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
        logger=eval_logger,
        observers=experiment.observers)

  steps = 0
  while steps < experiment.max_number_of_steps:
    if eval_loop:
      eval_loop.run(num_episodes=num_eval_episodes)
    train_loop.run(num_steps=eval_every)
    steps += eval_every
  if eval_loop:
    eval_loop.run(num_episodes=num_eval_episodes)


class _TrainingAdder(acme.adders.base.Adder):
  """Experience adder which executes training when there is sufficient data."""

  def __init__(self, adder: acme.adders.base.Adder, learner: core.Learner,
               iterator: core.PrefetchingIterator,
               replay_tables: Sequence[reverb.Table],
               max_diffs: Sequence[float]):
    """Initializes _TrainingAdder.

    Args:
      adder: Underlying, to be wrapped, adder used to add experience to Reverb.
      learner: Learner on which step() is to be called when there is data.
      iterator: Iterator used by the Learner to fetch training data.
      replay_tables: Collection of tables from which Learner fetches data
        through the iterator.
      max_diffs: Corresponding max_diff settings of the original rate limiters
        (before the _disable_insert_blocking call) corresponding to the
        `replay_tables`.
    """
    self._adder = adder
    self._learner = learner
    self._iterator = iterator
    self._replay_tables = replay_tables
    self._max_diffs = max_diffs
    self._learner_steps = 0

  def add_first(self, timestep: dm_env.TimeStep):
    self._maybe_train()
    self._adder.add_first(timestep)

  def add(self,
          action: types.NestedArray,
          next_timestep: dm_env.TimeStep,
          extras: types.NestedArray = ()):
    self._maybe_train()
    self._adder.add(action, next_timestep, extras)

  def reset(self):
    self._adder.reset()

  def _maybe_train(self):
    while True:
      if self._iterator.ready():
        self._learner.step()
        batches = self._iterator.retrieved_elements() - self._learner_steps
        self._learner_steps += 1
        assert batches == 1, (
            'Learner step must retrieve exactly one element from the iterator'
            f' (retrieved {batches}). Otherwise agent can deadlock.')
      elif self._learner_steps == 0:
        # Make sure `min_size_to_sample_` was reached before checking
        # `max_diff`.
        return
      else:
        # If any of the rate limiters would block the insert call, we sleep
        # a bit to allow for Learner's iterator to fetch data from the table.
        # As a result, either making a Learner step should be possible or insert
        # call won't be blocked anymore due to some data being sampled from the
        # table in the background.
        can_insert = True
        for table, max_diff in zip(self._replay_tables, self._max_diffs):
          info = table.info.rate_limiter_info
          diff = (
              info.insert_stats.completed * info.samples_per_insert -
              info.sample_stats.completed)
          if diff > max_diff:
            can_insert = False
        if can_insert:
          return
        else:
          time.sleep(0.001)


def _disable_insert_blocking(
    tables: Sequence[reverb.Table]
) -> Tuple[Sequence[reverb.Table], Sequence[float]]:
  """Disables blocking of insert operations for a given collection of tables."""
  modified_tables = []
  max_diffs = []
  for table in tables:
    rate_limiter_info = table.info.rate_limiter_info
    rate_limiter = reverb.rate_limiters.RateLimiter(
        samples_per_insert=rate_limiter_info.samples_per_insert,
        min_size_to_sample=rate_limiter_info.min_size_to_sample,
        min_diff=rate_limiter_info.min_diff,
        max_diff=sys.float_info.max)
    modified_tables.append(table.replace(rate_limiter=rate_limiter))
    max_diffs.append(rate_limiter_info.max_diff)
  return modified_tables, max_diffs
