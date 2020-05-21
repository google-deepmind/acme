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

"""D4PG agent implementation."""

import copy

from acme import datasets
from acme import networks
from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.agents import actors_tf2 as actors
from acme.agents import agent
from acme.agents.d4pg import learning
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tf2_utils

import reverb
import sonnet as snt
import tensorflow as tf


# TODO(b/145531941): make the naming of this agent consistent.
class D4PG(agent.Agent):
  """D4PG Agent.

  This implements a single-process D4PG agent. This is an actor-critic algorithm
  that generates data via a behavior policy, inserts N-step transitions into
  a replay buffer, and periodically updates the policy (and as a result the
  behavior) by sampling uniformly from this buffer.
  """

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               policy_network: snt.Module,
               critic_network: snt.Module,
               observation_network: types.TensorTransformation = tf.identity,
               discount: float = 0.99,
               batch_size: int = 256,
               prefetch_size: int = 4,
               target_update_period: int = 100,
               min_replay_size: int = 1000,
               max_replay_size: int = 1000000,
               samples_per_insert: float = 32.0,
               n_step: int = 5,
               sigma: float = 0.3,
               clipping: bool = True,
               logger: loggers.Logger = None,
               counter: counting.Counter = None,
               checkpoint: bool = True,
               replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      policy_network: the online (optimized) policy.
      critic_network: the online critic.
      observation_network: optional network to transform the observations before
        they are fed into any network.
      discount: discount to use for TD updates.
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      min_replay_size: minimum replay size before updating.
      max_replay_size: maximum replay size.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      n_step: number of steps to squash into a single transition.
      sigma: standard deviation of zero-mean, Gaussian exploration noise.
      clipping: whether to clip gradients by global norm.
      logger: logger object to be used by learner.
      counter: counter object used to keep track of steps.
      checkpoint: boolean indicating whether to checkpoint the learner.
      replay_table_name: string indicating what name to give the replay table.
    """
    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        priority_fns={replay_table_name: lambda x: 1.},
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(
        table=replay_table_name,
        client=reverb.TFClient(address),
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        environment_spec=environment_spec,
        transition_adder=True)

    # Make sure observation network is a Sonnet Module.
    observation_network = tf2_utils.to_sonnet_module(observation_network)

    # Create target networks.
    target_policy_network = copy.deepcopy(policy_network)
    target_critic_network = copy.deepcopy(critic_network)
    target_observation_network = copy.deepcopy(observation_network)

    # Get observation and action specs.
    act_spec = environment_spec.actions
    obs_spec = environment_spec.observations
    emb_spec = tf2_utils.create_variables(observation_network, [obs_spec])

    # Create the behavior policy.
    behavior_network = snt.Sequential([
        observation_network,
        policy_network,
        networks.ClippedGaussian(sigma),
        networks.ClipToSpec(act_spec),
    ])

    # Create variables.
    tf2_utils.create_variables(policy_network, [emb_spec])
    tf2_utils.create_variables(critic_network, [emb_spec, act_spec])
    tf2_utils.create_variables(target_policy_network, [emb_spec])
    tf2_utils.create_variables(target_critic_network, [emb_spec, act_spec])
    tf2_utils.create_variables(target_observation_network, [obs_spec])

    # Create the actor which defines how we take actions.
    actor = actors.FeedForwardActor(behavior_network, adder=adder)

    # Create optimizers.
    policy_optimizer = snt.optimizers.Adam(learning_rate=1e-4)
    critic_optimizer = snt.optimizers.Adam(learning_rate=1e-4)

    # The learner updates the parameters (and initializes them).
    learner = learning.D4PGLearner(
        policy_network=policy_network,
        critic_network=critic_network,
        observation_network=observation_network,
        target_policy_network=target_policy_network,
        target_critic_network=target_critic_network,
        target_observation_network=target_observation_network,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        clipping=clipping,
        discount=discount,
        target_update_period=target_update_period,
        dataset=dataset,
        counter=counter,
        logger=logger,
        checkpoint=checkpoint,
    )

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)
