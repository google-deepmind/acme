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

"""Example running D4PG on the OpenAI Gym."""

from typing import Mapping, Sequence

from absl import app
from absl import flags
import acme
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
import dm_env
import gym
import numpy as np
import sonnet as snt

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 100,
                     'Number of training episodes to run for.')
flags.DEFINE_integer('num_episodes_per_eval', 10,
                     'Number of training episodes to run between evaluation '
                     'episodes.')


def make_environment(
    task: str = 'MountainCarContinuous-v0') -> dm_env.Environment:
  """Creates an OpenAI Gym environment."""

  # Load the gym environment.
  environment = gym.make(task)

  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment


# The default settings in this network factory will work well for the
# MountainCarContinuous-v0 task but may need to be tuned for others. In
# particular, the vmin/vmax and num_atoms hyperparameters should be set to
# give the distributional critic a good dynamic range over possible discounted
# returns. Note that this is very different than the scale of immediate rewards.
def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
  """Creates the networks used by the agent."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(action_spec.shape, dtype=int)

  # Create the shared observation network; here simply a state-less operation.
  observation_network = tf2_utils.batch_concat

  # Create the policy network.
  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.NearZeroInitializedLinear(num_dimensions),
      networks.TanhToSpec(action_spec),
  ])

  # Create the critic network.
  critic_network = snt.Sequential([
      # The multiplexer concatenates the observations/actions.
      networks.CriticMultiplexer(),
      networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
      networks.DiscreteValuedHead(vmin, vmax, num_atoms),
  ])

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': observation_network,
  }


def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = make_environment()
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = make_networks(environment_spec.actions)

  # Construct the agent.
  agent = d4pg.D4PG(
      environment_spec=environment_spec,
      policy_network=agent_networks['policy'],
      critic_network=agent_networks['critic'],
      observation_network=agent_networks['observation'],
      sigma=1.0,  # pytype: disable=wrong-arg-types
  )

  # Create the environment loop used for training.
  train_loop = acme.EnvironmentLoop(environment, agent, label='train_loop')

  # Create the evaluation policy.
  eval_policy = snt.Sequential([
      agent_networks['observation'],
      agent_networks['policy'],
  ])

  # Create the evaluation actor and loop.
  eval_actor = actors.FeedForwardActor(policy_network=eval_policy)
  eval_env = make_environment()
  eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop')

  for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
    train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)
    eval_loop.run(num_episodes=1)


if __name__ == '__main__':
  app.run(main)
