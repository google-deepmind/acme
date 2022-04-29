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

"""Tests for local and distributed SAC-fd agents using layouts."""

import acme
from acme import specs
from acme.agents.jax import sac
from acme.agents.jax.lfd import config
from acme.agents.jax.lfd import sacfd_agents
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


class SACfDTest(absltest.TestCase):

  def test_sac_fd(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, action_dim=3, observation_dim=5, bounded=True)
    spec = specs.make_environment_spec(environment)

    # Create the networks.
    sac_network = sac.make_networks(spec)

    batch_size = 10
    sac_config = sac.SACConfig(
        batch_size=batch_size,
        target_entropy=sac.target_entropy_from_env_spec(spec),
        min_replay_size=1,
        samples_per_insert_tolerance_rate=2.0)
    lfd_config = config.LfdConfig(
        initial_insert_count=0, demonstration_ratio=0.2)
    sac_fd_config = sacfd_agents.SACfDConfig(
        lfd_config=lfd_config, sac_config=sac_config)
    counter = counting.Counter()
    agent = sacfd_agents.SACfD(
        spec=spec,
        sac_network=sac_network,
        sac_fd_config=sac_fd_config,
        lfd_iterator_fn=fake_demonstration_iterator,
        seed=0,
        counter=counter)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    loop.run(num_episodes=20)


class DistributedSACfDTest(absltest.TestCase):

  def test_distributed_sac_fd(self):

    def make_env(seed):
      del seed
      return fakes.ContinuousEnvironment(
          episode_length=10, action_dim=3, observation_dim=5, bounded=True)

    spec = specs.make_environment_spec(make_env(seed=0))

    batch_size = 10
    sac_config = sac.SACConfig(
        batch_size=batch_size,
        target_entropy=sac.target_entropy_from_env_spec(spec),
        min_replay_size=16,
        samples_per_insert=2)
    lfd_config = config.LfdConfig(
        initial_insert_count=0, demonstration_ratio=0.2)
    sac_fd_config = sacfd_agents.SACfDConfig(
        lfd_config=lfd_config, sac_config=sac_config)

    agent = sacfd_agents.DistributedSACfD(
        environment_factory=make_env,
        network_factory=sac.make_networks,
        sac_fd_config=sac_fd_config,
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
