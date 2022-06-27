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

"""The MPPI-inspired JAX actor."""

from typing import List, Mapping, Optional, Tuple

from acme import adders
from acme import core
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax.mbop import models
from acme.agents.jax.mbop import mppi
from acme.agents.jax.mbop import networks as mbop_networks
from acme.jax import networks as networks_lib
from acme.jax import running_statistics
from acme.jax import variable_utils
import jax
from jax import numpy as jnp

# Recurrent state is the trajectory.
Trajectory = jnp.ndarray

ActorCore = actor_core_lib.ActorCore[
    actor_core_lib.SimpleActorCoreRecurrentState[Trajectory],
    Mapping[str, jnp.ndarray]]


def make_actor_core(
    mppi_config: mppi.MPPIConfig,
    world_model: models.WorldModel,
    policy_prior: models.PolicyPrior,
    n_step_return: models.NStepReturn,
    environment_spec: specs.EnvironmentSpec,
    mean_std: Optional[running_statistics.NestedMeanStd] = None,
) -> ActorCore:
  """Creates an actor core wrapping the MBOP-configured MPPI planner.

  Args:
    mppi_config: Planner hyperparameters.
    world_model: A world model.
    policy_prior: A policy prior.
    n_step_return: An n-step return.
    environment_spec: Used to initialize the initial trajectory data structure.
    mean_std: Used to undo normalization if the networks trained normalized.

  Returns:
    A recurrent actor core.
  """

  if mean_std is not None:
    mean_std_observation = running_statistics.NestedMeanStd(
        mean=mean_std.mean.observation, std=mean_std.std.observation)
    mean_std_action = running_statistics.NestedMeanStd(
        mean=mean_std.mean.action, std=mean_std.std.action)
    mean_std_reward = running_statistics.NestedMeanStd(
        mean=mean_std.mean.reward, std=mean_std.std.reward)
    mean_std_n_step_return = running_statistics.NestedMeanStd(
        mean=mean_std.mean.extras['n_step_return'],
        std=mean_std.std.extras['n_step_return'])

    def denormalized_world_model(
        params: networks_lib.Params, observation_t: networks_lib.Observation,
        action_t: networks_lib.Action
    ) -> Tuple[networks_lib.Observation, networks_lib.Value]:
      """Denormalizes the reward for proper weighting in the planner."""
      observation_tp1, normalized_reward_t = world_model(
          params, observation_t, action_t)
      reward_t = running_statistics.denormalize(normalized_reward_t,
                                                mean_std_reward)
      return observation_tp1, reward_t

    planner_world_model = denormalized_world_model

    def denormalized_n_step_return(
        params: networks_lib.Params, observation_t: networks_lib.Observation,
        action_t: networks_lib.Action) -> networks_lib.Value:
      """Denormalize the n-step return for proper weighting in the planner."""
      normalized_n_step_return_t = n_step_return(params, observation_t,
                                                 action_t)
      return running_statistics.denormalize(normalized_n_step_return_t,
                                            mean_std_n_step_return)

    planner_n_step_return = denormalized_n_step_return
  else:
    planner_world_model = world_model
    planner_n_step_return = n_step_return

  def recurrent_policy(
      params_list: List[networks_lib.Params],
      random_key: networks_lib.PRNGKey,
      observation: networks_lib.Observation,
      previous_trajectory: Trajectory,
  ) -> Tuple[networks_lib.Action, Trajectory]:
    # Note that splitting the random key is handled by GenericActor.
    if mean_std is not None:
      observation = running_statistics.normalize(
          observation, mean_std=mean_std_observation)
    trajectory = mppi.mppi_planner(
        config=mppi_config,
        world_model=planner_world_model,
        policy_prior=policy_prior,
        n_step_return=planner_n_step_return,
        world_model_params=params_list[0],
        policy_prior_params=params_list[1],
        n_step_return_params=params_list[2],
        random_key=random_key,
        observation=observation,
        previous_trajectory=previous_trajectory)
    action = trajectory[0, ...]
    if mean_std is not None:
      action = running_statistics.denormalize(action, mean_std=mean_std_action)
    return (action, trajectory)

  batched_policy = jax.vmap(recurrent_policy, in_axes=(None, None, 0, 0))
  batched_policy = jax.jit(batched_policy)

  initial_trajectory = mppi.get_initial_trajectory(
      config=mppi_config, env_spec=environment_spec)
  initial_trajectory = jnp.expand_dims(initial_trajectory, axis=0)

  return actor_core_lib.batched_recurrent_to_actor_core(batched_policy,
                                                        initial_trajectory)


def make_ensemble_actor_core(
    networks: mbop_networks.MBOPNetworks,
    mppi_config: mppi.MPPIConfig,
    environment_spec: specs.EnvironmentSpec,
    mean_std: Optional[running_statistics.NestedMeanStd] = None,
    use_round_robin: bool = True,
) -> ActorCore:
  """Creates an actor core that uses ensemble models.

  Args:
    networks: MBOP networks.
    mppi_config: Planner hyperparameters.
    environment_spec: Used to initialize the initial trajectory data structure.
    mean_std: Used to undo normalization if the networks trained normalized.
    use_round_robin: Whether to use round robin or mean to calculate the policy
      prior over the ensemble members.

  Returns:
    A recurrent actor core.
  """
  world_model = models.make_ensemble_world_model(networks.world_model_network)
  policy_prior = models.make_ensemble_policy_prior(
      networks.policy_prior_network,
      environment_spec,
      use_round_robin=use_round_robin)
  n_step_return = models.make_ensemble_n_step_return(
      networks.n_step_return_network)

  return make_actor_core(mppi_config, world_model, policy_prior, n_step_return,
                         environment_spec, mean_std)


def make_actor(actor_core: ActorCore,
               random_key: networks_lib.PRNGKey,
               variable_source: core.VariableSource,
               adder: Optional[adders.Adder] = None) -> core.Actor:
  """Creates an MBOP actor from an actor core.

  Args:
    actor_core: An MBOP actor core.
    random_key: JAX Random key.
    variable_source: The source to get networks parameters from.
    adder: An adder to add experiences to. The `extras` of the adder holds the
      state of the recurrent policy. If `has_extras=True` then the `extras` part
      returned from the recurrent policy is appended to the state before added
      to the adder.

  Returns:
    A recurrent actor.
  """
  variable_client = variable_utils.VariableClient(
      client=variable_source,
      key=['world_model-policy', 'policy_prior-policy', 'n_step_return-policy'])

  return actors.GenericActor(
      actor_core, random_key, variable_client, adder, backend=None)
