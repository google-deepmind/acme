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

"""Tests for the PWIL agent."""

import acme
from acme import specs
from acme import types
from acme.agents.jax import pwil
from acme.agents.jax import sac
from acme.testing import fakes
from acme.utils import counting

from absl.testing import absltest
from absl.testing import parameterized


class PwilTest(parameterized.TestCase):

  def test_pwil(self):
    episode_length = 1
    num_transitions_rb = 100
    environment = fakes.ContinuousEnvironment(
        episode_length=episode_length,
        action_dim=3,
        observation_dim=5,
        bounded=True)
    # fakes.transition_dataset does not generate episodic datasets. Produce
    # length-1 episodes by replacing the environment discount with 0. The odd
    # number of discount=0 transitions in dataset_demonstration leads to an edge
    # case: length-0 episode at the end.
    dataset_demonstration = fakes.transition_dataset(environment).take(5)
    dataset_demonstration = dataset_demonstration.map(
        lambda sample: types.Transition(*sample.data)._replace(discount=0.))
    demonstrations_fn = (
        lambda: pwil.PWILDemonstrations(dataset_demonstration, 1))
    spec = specs.make_environment_spec(environment)
    pwil_config = pwil.PWILConfig(num_transitions_rb=num_transitions_rb,)

    networks = sac.make_networks(spec=spec)
    sac_config = sac.SACConfig(
        samples_per_insert_tolerance_rate=2.0, min_replay_size=1)
    rl_agent = sac.SACBuilder(config=sac_config)
    behavior_policy = sac.apply_policy_and_sample(networks)

    counter = counting.Counter()
    # Construct the agent. Setup learning batches low enough to trigger updates.
    agent = pwil.PWIL(
        spec=spec,
        rl_agent=rl_agent,
        config=pwil_config,
        networks=networks,
        seed=0,
        batch_size=8,
        demonstrations_fn=demonstrations_fn,
        policy_network=behavior_policy,
        counter=counter)

    train_loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    train_loop.run(num_episodes=3)


if __name__ == '__main__':
  absltest.main()
