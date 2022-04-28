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
from typing import Optional

import acme
from acme import core
from acme import specs
from acme import types
from acme.agents.jax import builders
from acme.jax import utils
from acme.utils import counting
from acme.utils import experiment_utils
import dm_env
import jax
import reverb


class _LearningActor(core.Actor):
  """Actor which learns (updates its parameters) when `update` is called.

  This combines a base actor, a learner, and an iterator over data. Whenever
  `update` is called on the wrapping actor the learner will take a step
  (e.g. one step of gradient descent) until data from the iterator has been
  consumed. Selecting actions and making observations are handled by the base
  actor.
  Intended to be used by the `run_agent` only.
  """

  def __init__(self, actor: core.Actor, learner: core.Learner,
               iterator: Optional[core.PrefetchingIterator] = None):
    self._actor = actor
    self._learner = learner
    self._iterator = iterator

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    return self._actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    self._actor.observe(action, next_timestep)

  def update(self):
    # Perform learner steps as long as iterator has data.
    update_actor = False
    while self._iterator.ready():
      # Run learner steps (usually means gradient steps).
      self._learner.step()
      update_actor = True
    if update_actor:
      # Update the actor weights only when learner was updated.
      self._actor.update()


def disable_insert_blocking(table: reverb.Table):
  rate_limiter_info = table.info.rate_limiter_info
  rate_limiter = reverb.rate_limiters.RateLimiter(
      samples_per_insert=rate_limiter_info.samples_per_insert,
      min_size_to_sample=rate_limiter_info.min_size_to_sample,
      min_diff=rate_limiter_info.min_diff,
      max_diff=sys.float_info.max)
  return table.replace(rate_limiter=rate_limiter)


# TODO(stanczyk): Align interface of this function with distributed program.
def run_agent(builder: builders.ActorLearnerBuilder,
              environment: dm_env.Environment,
              networks,
              policy_network,
              eval_policy_network,
              seed: int = 0,
              num_steps: int = 1_000,
              eval_every: int = 100,
              num_eval_episodes: int = 1):
  """Runs training with evaluation of a given agent."""

  key = jax.random.PRNGKey(seed)

  # Create the replay server and grab its address.
  environment_spec = specs.make_environment_spec(environment)
  replay_tables = builder.make_replay_tables(environment_spec)

  # Disable blocking of inserts by tables' rate limiters, as this function
  # executes learning (sampling from the table) and data generation
  # (inserting into the table) sequentially from the same thread
  # which could result in blocked insert making the algorithm hang.
  replay_tables = [disable_insert_blocking(table) for table in replay_tables]

  replay_server = reverb.Server(replay_tables, port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')

  # Create actor, dataset, and learner for generating, storing, and consuming
  # data respectively.
  adder = builder.make_adder(replay_client)

  dataset = builder.make_dataset_iterator(replay_client)
  # We always use prefetch, as it provides an iterator with additional
  # 'ready' method.
  dataset = utils.prefetch(dataset, buffer_size=1)
  learner_key, key = jax.random.split(key)
  learner = builder.make_learner(
      random_key=learner_key,
      networks=networks,
      dataset=dataset,
      replay_client=replay_client)

  actor_key, key = jax.random.split(key)
  actor = builder.make_actor(
      actor_key, policy_network, adder, variable_source=learner)

  # Create the environment loop used for training.
  train_logger = experiment_utils.make_experiment_logger(
      label='train', steps_key='train_steps')

  # Replace the actor with a LearningActor. This makes sure that every time
  # that `update` is called on the actor it checks to see whether there is
  # any new data to learn from and if so it runs a learner step. The rate
  # at which new data is released is controlled by the replay table's
  # rate_limiter which is created by the builder.make_replay_tables call above.
  actor = _LearningActor(actor, learner, dataset)

  train_loop = acme.EnvironmentLoop(
      environment,
      actor,
      counter=counting.Counter(prefix='train'),
      logger=train_logger)

  # Create the evaluation actor and loop.
  eval_logger = experiment_utils.make_experiment_logger(
      label='eval', steps_key='eval_steps')
  eval_actor = builder.make_actor(
      random_key=jax.random.PRNGKey(seed),
      policy_network=eval_policy_network,
      variable_source=learner)
  eval_loop = acme.EnvironmentLoop(
      environment,
      eval_actor,
      counter=counting.Counter(prefix='eval'),
      logger=eval_logger)

  assert num_steps % eval_every == 0
  for _ in range(num_steps // eval_every):
    eval_loop.run(num_episodes=num_eval_episodes)
    train_loop.run(num_steps=eval_every)
  eval_loop.run(num_episodes=num_eval_episodes)
