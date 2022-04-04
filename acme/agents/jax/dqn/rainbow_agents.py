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

"""Defines distributed and local Rainbow agents, using JAX."""

import dataclasses
import functools
from typing import Callable, Optional, Sequence

from acme import specs
from acme.agents.jax.dqn import actor as dqn_actor
from acme.agents.jax.dqn import builder
from acme.agents.jax.dqn import config as dqn_config
from acme.agents.jax.dqn import losses
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import loggers
import dm_env
import rlax

NetworkFactory = Callable[[specs.EnvironmentSpec],
                          networks_lib.FeedForwardNetwork]


@dataclasses.dataclass
class RainbowConfig(dqn_config.DQNConfig):
  """(Additional) configuration options for RainbowDQN."""
  max_abs_reward: float = 1.0  # For clipping reward


def apply_policy_and_sample(network: networks_lib.FeedForwardNetwork,
                            ) -> dqn_actor.EpsilonPolicy:
  """Returns a function that computes actions.

  Note that this differs from default_behavior_policy with that it
  expects c51-style network head which returns a tuple with the first entry
  representing q-values.

  Args:
    network: A c51-style feedforward network.

  Returns:
    A feedforward policy.
  """

  def apply_and_sample(params, key, obs, epsilon):
    # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
    obs = utils.add_batch_dim(obs)
    action_values = network.apply(params, obs)[0]
    action_values = utils.squeeze_batch_dim(action_values)
    return rlax.epsilon_greedy(epsilon).sample(key, action_values)

  return apply_and_sample


def eval_policy(network: networks_lib.FeedForwardNetwork, eval_epsilon: float
                ) -> dqn_actor.EpsilonPolicy:
  """Returns a function that computes actions.

  Note that this differs from default_behavior_policy with that it
  expects c51-style network head which returns a tuple with the first entry
  representing q-values.

  Args:
    network: A c51-style feedforward network.
    eval_epsilon: for epsilon-greedy exploration.

  Returns:
    A feedforward policy.
  """
  policy = apply_policy_and_sample(network)

  def apply_and_sample(params, key, obs, _):
    return policy(params, key, obs, eval_epsilon)

  return apply_and_sample


def make_builder(
    config: RainbowConfig,
    logger_fn: Callable[[], loggers.Logger] = lambda: None,
):
  """Returns a DQNBuilder with a pre-built loss function."""
  loss_fn = losses.PrioritizedCategoricalDoubleQLearning(
      discount=config.discount,
      importance_sampling_exponent=config.importance_sampling_exponent,
      max_abs_reward=config.max_abs_reward,
  )
  return builder.DQNBuilder(config, loss_fn=loss_fn, logger_fn=logger_fn)


class DistributedRainbow(distributed_layout.DistributedLayout):
  """Distributed program definition for Rainbow."""

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      network_factory: NetworkFactory,
      config: RainbowConfig,
      seed: int,
      num_actors: int,
      max_number_of_steps: Optional[int] = None,
      log_to_bigtable: bool = False,
      log_every: float = 10.0,
      eval_epsilon: float = 0.0,
      evaluator_factories: Optional[Sequence[
          distributed_layout.EvaluatorFactory]] = None,
  ):
    logger_fn = functools.partial(
        loggers.make_default_logger,
        'learner',
        log_to_bigtable,
        time_delta=log_every,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')
    dqn_builder = make_builder(config, logger_fn=logger_fn)
    train_policy_factory = apply_policy_and_sample
    if evaluator_factories is None:
      eval_policy_factory = lambda n: eval_policy(n, eval_epsilon)
      evaluator_factories = [
          distributed_layout.default_evaluator_factory(
              environment_factory=lambda seed: environment_factory(True),
              network_factory=network_factory,
              policy_factory=eval_policy_factory,
              log_to_bigtable=log_to_bigtable)
      ]
    super().__init__(
        seed=seed,
        environment_factory=lambda seed: environment_factory(False),
        network_factory=network_factory,
        builder=dqn_builder,
        policy_network=train_policy_factory,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every))


class RainbowDQN(local_layout.LocalLayout):
  """Local agent for Rainbow."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: networks_lib.FeedForwardNetwork,
      config: RainbowConfig,
      seed: int,
  ):
    min_replay_size = config.min_replay_size
    # Local layout (actually agent.Agent) makes sure that we populate the
    # buffer with min_replay_size initial transitions and that there's no need
    # for tolerance_rate. In order for deadlocks not to happen we need to
    # disable rate limiting that heppens inside the DQNBuilder. This is achieved
    # by the following two lines.
    config.samples_per_insert_tolerance_rate = float('inf')
    config.min_replay_size = 1
    self.builder = make_builder(config)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=apply_policy_and_sample(network),
        batch_size=config.batch_size,
        samples_per_insert=config.samples_per_insert,
        min_replay_size=min_replay_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
    )
