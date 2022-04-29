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

"""Tests for local and distributed TD3fD agents using layouts."""

import acme
from acme import specs
from acme.agents.jax import lfd
from acme.agents.jax import td3
from acme.testing import fakes
from acme.utils import counting
import dm_env
import launchpad as lp
import numpy as np

from absl.testing import absltest


def fake_demonstration_iterator():
  k = 0
  while True:
    action = np.random.uniform(low=0., high=1., size=3).astype(np.float32)
    obs = np.random.uniform(low=0., high=1., size=5).astype(np.float32)
    reward = np.float32(0.)
    discount = np.float32(0.)
    if k % 10 == 0:
      ts = dm_env.restart(obs)
    elif k % 10 == 9:
      ts = dm_env.TimeStep(dm_env.StepType.LAST, reward, discount, obs)
    else:
      ts = dm_env.transition(reward=reward, observation=obs, discount=discount)
    k += 1
    yield action, ts


class TD3fDTest(absltest.TestCase):

  def test_td3_fd(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, action_dim=3, observation_dim=5, bounded=True)
    spec = specs.make_environment_spec(environment)

    # Create the networks.
    td3_network = td3.make_networks(spec)

    batch_size = 10
    td3_config = td3.TD3Config(
        batch_size=batch_size,
        min_replay_size=1,
        samples_per_insert_tolerance_rate=2.0)
    lfd_config = lfd.LfdConfig(initial_insert_count=0, demonstration_ratio=0.2)
    td3_fd_config = lfd.TD3fDConfig(
        lfd_config=lfd_config, td3_config=td3_config)
    counter = counting.Counter()
    agent = lfd.TD3fD(
        spec=spec,
        td3_network=td3_network,
        td3_fd_config=td3_fd_config,
        lfd_iterator_fn=fake_demonstration_iterator,
        seed=0,
        counter=counter)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    loop.run(num_episodes=20)


class DistributedTD3fDTest(absltest.TestCase):

  def test_distributed_td3_fd(self):

    def make_env(seed):
      del seed
      return fakes.ContinuousEnvironment(
          episode_length=10, action_dim=3, observation_dim=5, bounded=True)

    batch_size = 10
    td3_config = td3.TD3Config(
        batch_size=batch_size, min_replay_size=16, samples_per_insert=2)
    lfd_config = lfd.LfdConfig(initial_insert_count=0, demonstration_ratio=0.2)
    td3_fd_config = lfd.TD3fDConfig(
        lfd_config=lfd_config, td3_config=td3_config)

    spec = specs.make_environment_spec(make_env(0))

    agent = lfd.DistributedTD3fD(
        environment_factory=make_env,
        environment_spec=spec,
        network_factory=td3.make_networks,
        td3_fd_config=td3_fd_config,
        lfd_iterator_fn=fake_demonstration_iterator,
        seed=0,
        num_actors=2)

    program = agent.build()
    (learner_node,) = program.groups['learner']
    learner_node.disable_run()  # pytype: disable=attribute-error

    lp.launch(program, launch_type='test_mt')

    learner: acme.Learner = learner_node.create_handle().dereference()

    for _ in range(5):
      learner.step()


if __name__ == '__main__':
  absltest.main()
