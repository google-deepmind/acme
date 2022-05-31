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

"""Tests for mppi."""
import functools
from typing import Any

from acme import specs
from acme.agents.jax.mbop import ensemble
from acme.agents.jax.mbop import models
from acme.agents.jax.mbop import mppi
from acme.jax import networks as networks_lib
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


def get_fake_world_model() -> networks_lib.FeedForwardNetwork:

  def apply(params: Any, observation_t: jnp.ndarray, action_t: jnp.ndarray):
    del params
    return observation_t, jnp.ones((
        action_t.shape[0],
        1,
    ))

  return networks_lib.FeedForwardNetwork(init=lambda: None, apply=apply)


def get_fake_policy_prior() -> networks_lib.FeedForwardNetwork:
  return networks_lib.FeedForwardNetwork(
      init=lambda: None,
      apply=lambda params, observation_t, action_tm1: action_tm1)


def get_fake_n_step_return() -> networks_lib.FeedForwardNetwork:

  def apply(params, observation_t, action_t):
    del params, action_t
    return jnp.ones((observation_t.shape[0], 1))

  return networks_lib.FeedForwardNetwork(init=lambda: None, apply=apply)


class WeightedAverageTests(parameterized.TestCase):

  @parameterized.parameters((np.array([1, 1, 1]), 1), (np.array([0, 1, 0]), 10),
                            (np.array([-1, 1, -1]), 4),
                            (np.array([-10, 30, 0]), -0.5))
  def test_weighted_averages(self, cum_reward, kappa):
    """Compares method with a local version of the exp-weighted averaging."""
    action_trajectories = jnp.reshape(
        jnp.arange(3 * 10 * 4), (3, 10, 4), order='F')
    averaged_trajectory = mppi.return_weighted_average(
        action_trajectories=action_trajectories,
        cum_reward=cum_reward,
        kappa=kappa)
    exp_weights = jnp.exp(kappa * cum_reward)
    # Verify single-value averaging lines up with the global averaging call:
    for i in range(10):
      for j in range(4):
        np.testing.assert_allclose(
            averaged_trajectory[i, j],
            jnp.sum(exp_weights * action_trajectories[:, i, j]) /
            jnp.sum(exp_weights),
            atol=1E-5,
            rtol=1E-5)


class MPPITest(parameterized.TestCase):
  """This tests the MPPI planner to make sure it is correctly rolling out.

  It does not check the actual performance of the planner, as this would be a
  bit more complicated to set up.
  """

  # TODO(dulacarnold): Look at how we can check this is actually finding an
  # optimal path through the model.

  def setUp(self):
    super().setUp()
    self.state_dims = 8
    self.action_dims = 4
    self.params = {
        'world': jnp.ones((3,)),
        'policy': jnp.ones((3,)),
        'value': jnp.ones((3,))
    }
    self.env_spec = specs.EnvironmentSpec(
        observations=specs.Array(shape=(self.state_dims,), dtype=float),
        actions=specs.Array(shape=(self.action_dims,), dtype=float),
        rewards=specs.Array(shape=(1,), dtype=float, name='reward'),
        discounts=specs.BoundedArray(
            shape=(), dtype=float, minimum=0., maximum=1., name='discount'))

  @parameterized.named_parameters(('NO-PLAN', 0), ('NORMAL', 10))
  def test_planner_init(self, horizon: int):
    world_model = get_fake_world_model()
    rr_world_model = functools.partial(ensemble.apply_round_robin,
                                       world_model.apply)
    policy_prior = get_fake_policy_prior()

    def _rr_policy_prior(params, key, observation_t, action_tm1):
      del key
      return ensemble.apply_round_robin(
          policy_prior.apply,
          params,
          observation_t=observation_t,
          action_tm1=action_tm1)

    rr_policy_prior = models.feed_forward_policy_prior_to_actor_core(
        _rr_policy_prior, jnp.zeros((1, self.action_dims)))

    n_step_return = get_fake_n_step_return()
    n_step_return = functools.partial(ensemble.apply_mean, n_step_return.apply)

    config = mppi.MPPIConfig(
        sigma=1,
        beta=0.2,
        horizon=horizon,
        n_trajectories=9,
        action_aggregation_fn=functools.partial(
            mppi.return_weighted_average, kappa=1))
    previous_trajectory = mppi.get_initial_trajectory(config, self.env_spec)
    key = jax.random.PRNGKey(0)
    for _ in range(5):
      previous_trajectory = mppi.mppi_planner(
          config,
          world_model=rr_world_model,
          policy_prior=rr_policy_prior,
          n_step_return=n_step_return,
          world_model_params=self.params,
          policy_prior_params=self.params,
          n_step_return_params=self.params,
          random_key=key,
          observation=jnp.ones(self.state_dims),
          previous_trajectory=previous_trajectory)


if __name__ == '__main__':
  absltest.main()
