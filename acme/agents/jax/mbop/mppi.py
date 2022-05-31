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

"""Provides the extended MPPI planner used in MBOP [https://arxiv.org/abs/2008.05556].

In this context, MPPI refers to Model-Predictive Path Integral control,
originally introduced in "Model Predictive Path Integral Control: From Theory to
Parallel Computation" Grady Williams, Andrew Aldrich and Evangelos A. Theodorou.

This is a modified implementation of MPPI that adds a policy prior and n-step
return extension as described in Algorithm 2 of "Model-Based Offline Planning"
[https://arxiv.org/abs/2008.05556].  Notation is taken from the paper.  This
planner can be 'degraded' to provide both 'basic' MPPI or PDDM-style
[https://arxiv.org/abs/1909.11652] planning by removing the n-step return,
providing a Gaussian policy prior, or single-head ensembles.
"""
import dataclasses
import functools
from typing import Callable, Optional

from acme import specs
from acme.agents.jax.mbop import models
from acme.jax import networks
import jax
from jax import random
import jax.numpy as jnp

# Function that takes (n_trajectories, horizon, action_dim) tensor of action
# trajectories and  (n_trajectories) vector of corresponding cumulative rewards,
# i.e. returns, for each trajectory as input and returns a single action
# trajectory.
ActionAggregationFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def return_weighted_average(action_trajectories: jnp.ndarray,
                            cum_reward: jnp.ndarray,
                            kappa: float) -> jnp.ndarray:
  r"""Calculates return-weighted average over all trajectories.

  This will calculate the return-weighted average over a set of trajectories as
  defined on l.17 of Alg. 2 in the MBOP paper:
  [https://arxiv.org/abs/2008.05556].

  Note: Clipping will be performed for `cum_reward` values > 80 to avoid NaNs.

  Args:
    action_trajectories: (n_trajectories, horizon, action_dim) tensor of action
      trajectories, corresponds to `A` in Alg. 2.
    cum_reward: (n_trajectories) vector of corresponding cumulative rewards
      (returns) for each trajectory. Corresponds to `\mathcal{R}` in Alg. 2.
    kappa: `\kappa` constant, changes the 'peakiness' of the exponential
      averaging.

  Returns:
    Single action trajectory corresponding to the return-weighted average of the
      trajectories.
  """
  # Substract maximum reward to avoid NaNs:
  cum_reward = cum_reward - cum_reward.max()
  # Remove the batch dimension of cum_reward allows for an implicit broadcast in
  # jnp.average:
  exp_cum_reward = jnp.exp(kappa * jnp.squeeze(cum_reward))
  return jnp.average(action_trajectories, weights=exp_cum_reward, axis=0)


def return_top_k_average(action_trajectories: jnp.ndarray,
                         cum_reward: jnp.ndarray,
                         k: int = 10) -> jnp.ndarray:
  r"""Calculates the top-k average over all trajectories.

  This will calculate the top-k average over a set of trajectories as
  defined in the POIR Paper:

  Note: top-k average is more numerically stable than the weighted average.

  Args:
    action_trajectories: (n_trajectories, horizon, action_dim) tensor of action
      trajectories.
    cum_reward: (n_trajectories) vector of corresponding cumulative rewards
      (returns) for each trajectory.
    k: the number of trajectories to average.

  Returns:
    Single action trajectory corresponding to the average of the k best
      trajectories.
  """
  top_k_trajectories = action_trajectories[jnp.argsort(
      jnp.squeeze(cum_reward))[-int(k):]]
  return jnp.mean(top_k_trajectories, axis=0)


@dataclasses.dataclass
class MPPIConfig:
  """Config dataclass for MPPI-style planning, used in mppi.py.

  These variables correspond to different parameters of `MBOP-Trajopt` as
  defined in MBOP [https://arxiv.org/abs/2008.05556] (Alg. 2).

  Attributes:
    sigma: Variance of action-additive noise.
    beta: Mixture parameter between old trajectory and new action.
    horizon: Planning horizon, corresponds to H in Alg. 2 line 8.
    n_trajectories: Number of trajectories used in `mppi_planner` to sample the
      best action. Corresponds to `N` in Alg. 2 line. 5.
    previous_trajectory_clip: Value to clip the previous_trajectory's actions
      to. Disabled if None.
    action_aggregation_fn: Function that aggregates action trajectories and
      returns a single action trajectory.
  """
  sigma: float = 0.8
  beta: float = 0.2
  horizon: int = 15
  n_trajectories: int = 1000
  previous_trajectory_clip: Optional[float] = None
  action_aggregation_fn: ActionAggregationFn = (
      functools.partial(return_weighted_average, kappa=0.5))


def get_initial_trajectory(config: MPPIConfig, env_spec: specs.EnvironmentSpec):
  """Returns the initial empty trajectory `T_0`."""
  return jnp.zeros((max(1, config.horizon),) + env_spec.actions.shape)


def _repeat_n(new_batch: int, data: jnp.ndarray) -> jnp.ndarray:
  """Create new batch dimension of size `new_batch` by repeating `data`."""
  return jnp.broadcast_to(data, (new_batch,) + data.shape)


def mppi_planner(
    config: MPPIConfig,
    world_model: models.WorldModel,
    policy_prior: models.PolicyPrior,
    n_step_return: models.NStepReturn,
    world_model_params: networks.Params,
    policy_prior_params: networks.Params,
    n_step_return_params: networks.Params,
    random_key: networks.PRNGKey,
    observation: networks.Observation,
    previous_trajectory: jnp.ndarray,
) -> jnp.ndarray:
  """MPPI-extended trajectory optimizer.

  This implements the trajectory optimizer described in MBOP
  [https://arxiv.org/abs/2008.05556] (Alg. 2) which is an extended version of
  MPPI that adds support for arbitrary sampling distributions and extends the
  return horizon using an approximate model of returns.  There are a couple
  notation changes for readability:
  A -> action_trajectories
  T -> action_trajectory

  If the horizon is set to 0, the planner will simply call the policy_prior
  and average the action over the ensemble heads.

  Args:
    config: Base configuration parameters of MPPI.
    world_model: Corresponds to `f_m(s_t, a_t)_s` in Alg. 2.
    policy_prior: Corresponds to `f_b(s_t, a_tm1)` in Alg. 2.
    n_step_return: Corresponds to `f_R(s_t, a_t)` in Alg. 2.
    world_model_params: Parameters for world model.
    policy_prior_params: Parameters for policy prior.
    n_step_return_params: Parameters for n_step return.
    random_key: JAX random key seed.
    observation: Normalized current observation from the environment, `s` in
      Alg. 2.
    previous_trajectory: Normalized previous action trajectory. `T` in Alg 2.
      Shape is [n_trajectories, horizon, action_dims].

  Returns:
    jnp.ndarray: Average action trajectory of shape [horizon, action_dims].
  """
  action_trajectory_tm1 = previous_trajectory
  policy_prior_state = policy_prior.init(random_key)

  # Broadcast so that we have n_trajectories copies of each:
  observation_t = jax.tree_map(
      functools.partial(_repeat_n, config.n_trajectories), observation)
  action_tm1 = jnp.broadcast_to(action_trajectory_tm1[0],
                                (config.n_trajectories,) +
                                action_trajectory_tm1[0].shape)

  if config.previous_trajectory_clip is not None:
    action_tm1 = jnp.clip(
        action_tm1,
        a_min=-config.previous_trajectory_clip,
        a_max=config.previous_trajectory_clip)

  # First check if planning is unnecessary:
  if config.horizon == 0:
    if hasattr(policy_prior_state, 'action_tm1'):
      policy_prior_state = policy_prior_state.replace(action_tm1=action_tm1)
    action_set, _ = policy_prior.select_action(policy_prior_params,
                                               observation_t,
                                               policy_prior_state)
    # Need to re-create an action trajectory from a single action.
    return jnp.broadcast_to(
        jnp.mean(action_set, axis=0), (1, action_set.shape[-1]))

  # Accumulators for returns and trajectories:
  cum_reward = jnp.zeros((config.n_trajectories, 1))

  # Generate noise once:
  random_key, noise_key = random.split(random_key)
  action_noise = config.sigma * random.normal(noise_key, (
      (config.horizon,) + action_tm1.shape))

  # Initialize empty set of action trajectories for concatenation in loop:
  action_trajectories = jnp.zeros((config.n_trajectories, 0) +
                                  action_trajectory_tm1[0].shape)

  for t in range(config.horizon):
    # Query policy prior for proposed action:
    if hasattr(policy_prior_state, 'action_tm1'):
      policy_prior_state = policy_prior_state.replace(action_tm1=action_tm1)
    action_t, policy_prior_state = policy_prior.select_action(
        policy_prior_params, observation_t, policy_prior_state)
    # Add action noise:
    action_t = action_t + action_noise[t]
    # Mix action with previous trajectory's corresponding action:
    action_t = (1 -
                config.beta) * action_t + config.beta * action_trajectory_tm1[t]

    # Query world model to get next observation and reward:
    observation_tp1, reward_t = world_model(world_model_params, observation_t,
                                            action_t)
    cum_reward += reward_t

    # Insert actions into trajectory matrix:
    action_trajectories = jnp.concatenate(
        [action_trajectories,
         jnp.expand_dims(action_t, axis=1)], axis=1)
    # Bump variable timesteps for next loop:
    observation_t = observation_tp1
    action_tm1 = action_t

  # De-normalize and append the final n_step return prediction:
  n_step_return_t = n_step_return(n_step_return_params, observation_t, action_t)
  cum_reward += n_step_return_t

  # Average the set of `n_trajectories` trajectories into a single trajectory.
  return config.action_aggregation_fn(action_trajectories, cum_reward)
