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

"""MBOP models."""

import functools
from typing import Callable, Generic, Optional, Tuple

from acme import specs
from acme.agents.jax import actor_core
from acme.agents.jax.mbop import ensemble
from acme.agents.jax.mbop import networks as mbop_networks
from acme.jax import networks
from acme.jax import utils
import chex
import jax

# World, policy prior and n-step return models. These are backed by the
# corresponding networks.
WorldModel = Callable[[networks.Params, networks.Observation, networks.Action],
                      Tuple[networks.Observation, networks.Value]]

PolicyPrior = actor_core.ActorCore

NStepReturn = Callable[[networks.Params, networks.Observation, networks.Action],
                       networks.Value]


@chex.dataclass(frozen=True, mappable_dataclass=False)
class PolicyPriorState(Generic[actor_core.RecurrentState]):
  """State of a policy prior.

  Attributes:
    rng: Random key.
    action_tm1: Previous action.
    recurrent_state: Recurrent state. It will be none for non-recurrent, e.g.
      feed forward, policies.
  """
  rng: networks.PRNGKey
  action_tm1: networks.Action
  recurrent_state: Optional[actor_core.RecurrentState] = None


FeedForwardPolicyState = PolicyPriorState[actor_core.NoneType]


def feed_forward_policy_prior_to_actor_core(
    policy: actor_core.RecurrentPolicy, initial_action_tm1: networks.Action
) -> actor_core.ActorCore[PolicyPriorState, actor_core.NoneType]:
  """A convenience adaptor from a feed forward policy prior to ActorCore.

  Args:
    policy: A feed forward policy prior. In the planner and other components,
      the previous action is explicitly passed as an argument to the policy
      prior together with the observation to infer the next action. Therefore,
      we model feed forward policy priors as recurrent ActorCore policies with
      previous action being the recurrent state.
    initial_action_tm1: Initial previous action. This will usually be a zero
      tensor.

  Returns:
    an ActorCore representing the feed forward policy prior.
  """

  def select_action(params: networks.Params, observation: networks.Observation,
                    state: FeedForwardPolicyState):
    rng, policy_rng = jax.random.split(state.rng)
    action = policy(params, policy_rng, observation, state.action_tm1)
    return action, PolicyPriorState(rng, action)

  def init(rng: networks.PRNGKey) -> FeedForwardPolicyState:
    return PolicyPriorState(rng, initial_action_tm1)

  def get_extras(unused_state: FeedForwardPolicyState) -> actor_core.NoneType:
    return None

  return actor_core.ActorCore(
      init=init, select_action=select_action, get_extras=get_extras)


def make_ensemble_world_model(
    world_model_network: mbop_networks.WorldModelNetwork) -> WorldModel:
  """Creates an ensemble world model from its network."""
  return functools.partial(ensemble.apply_round_robin,
                           world_model_network.apply)


def make_ensemble_policy_prior(
    policy_prior_network: mbop_networks.PolicyPriorNetwork,
    spec: specs.EnvironmentSpec,
    use_round_robin: bool = True) -> PolicyPrior:
  """Creates an ensemble policy prior from its network.

  Args:
    policy_prior_network: The policy prior network.
    spec: Environment specification.
    use_round_robin: Whether to use round robin or mean to calculate the policy
      prior over the ensemble members.

  Returns:
    A policy prior.
  """

  def _policy_prior(params: networks.Params, key: networks.PRNGKey,
                    observation_t: networks.Observation,
                    action_tm1: networks.Action) -> networks.Action:
    # Regressor policies are deterministic.
    del key
    apply_fn = (
        ensemble.apply_round_robin if use_round_robin else ensemble.apply_mean)
    return apply_fn(
        policy_prior_network.apply,
        params,
        observation_t=observation_t,
        action_tm1=action_tm1)

  dummy_action = utils.zeros_like(spec.actions)
  dummy_action = utils.add_batch_dim(dummy_action)

  return feed_forward_policy_prior_to_actor_core(_policy_prior, dummy_action)


def make_ensemble_n_step_return(
    n_step_return_network: mbop_networks.NStepReturnNetwork) -> NStepReturn:
  """Creates an ensemble n-step return model from its network."""
  return functools.partial(ensemble.apply_mean, n_step_return_network.apply)
