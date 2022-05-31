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

"""Tests for the MBOP agent."""

import functools

from acme import specs
from acme import types
from acme.agents.jax.mbop import learning
from acme.agents.jax.mbop import losses as mbop_losses
from acme.agents.jax.mbop import networks as mbop_networks
from acme.testing import fakes
from acme.utils import loggers
import chex
import jax
import optax
import rlds

from absl.testing import absltest


class MBOPTest(absltest.TestCase):

  def test_learner(self):
    with chex.fake_pmap_and_jit():
      num_sgd_steps_per_step = 1
      num_steps = 5
      num_networks = 7

      # Create a fake environment to test with.
      environment = fakes.ContinuousEnvironment(
          episode_length=10, bounded=True, observation_dim=3, action_dim=2)

      spec = specs.make_environment_spec(environment)
      dataset = fakes.transition_dataset(environment)

      # Add dummy n-step return to the transitions.
      def _add_dummy_n_step_return(sample):
        return types.Transition(*sample.data)._replace(
            extras={'n_step_return': 1.0})

      dataset = dataset.map(_add_dummy_n_step_return)
      # Convert into time-batched format with previous, current and next
      # transitions.
      dataset = rlds.transformations.batch(dataset, 3)
      dataset = dataset.batch(8).as_numpy_iterator()

      # Use the default networks and losses.
      networks = mbop_networks.make_networks(spec)
      losses = mbop_losses.MBOPLosses()

      def logger_fn(label: str, steps_key: str):
        return loggers.make_default_logger(label, steps_key=steps_key)

      def make_learner_fn(name, logger_fn, counter, rng_key, dataset, network,
                          loss):
        return learning.make_ensemble_regressor_learner(name, num_networks,
                                                        logger_fn, counter,
                                                        rng_key, dataset,
                                                        network, loss,
                                                        optax.adam(0.01),
                                                        num_sgd_steps_per_step)

      learner = learning.MBOPLearner(
          networks, losses, dataset, jax.random.PRNGKey(0), logger_fn,
          functools.partial(make_learner_fn, 'world_model'),
          functools.partial(make_learner_fn, 'policy_prior'),
          functools.partial(make_learner_fn, 'n_step_return'))

      # Train the agent
      for _ in range(num_steps):
        learner.step()

      # Save and restore.
      learner_state = learner.save()
      learner.restore(learner_state)


if __name__ == '__main__':
  absltest.main()
