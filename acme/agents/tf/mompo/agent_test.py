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

"""Tests for the multi-objective MPO agent."""

from typing import Dict, Sequence, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import acme
from acme import specs
from acme.agents.tf import mompo
from acme.testing import fakes
from acme.tf import networks
import numpy as np
import sonnet as snt
import tensorflow as tf


def make_networks(
    action_spec: specs.Array,
    num_critic_heads: int,
    policy_layer_sizes: Sequence[int] = (300, 200),
    critic_layer_sizes: Sequence[int] = (400, 300),
    num_layers_shared: int = 1,
    distributional_critic: bool = True,
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Dict[str, snt.Module]:
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          tanh_mean=False,
          init_scale=0.69)
  ])

  if not distributional_critic:
    critic_layer_sizes = list(critic_layer_sizes) + [1]

  if not num_layers_shared:
    # No layers are shared
    critic_network_base = None
  else:
    critic_network_base = networks.LayerNormMLP(
        critic_layer_sizes[:num_layers_shared], activate_final=True)
  critic_network_heads = [
      snt.nets.MLP(critic_layer_sizes, activation=tf.nn.elu,
                   activate_final=False)
      for _ in range(num_critic_heads)]
  if distributional_critic:
    critic_network_heads = [
        snt.Sequential([
            c, networks.DiscreteValuedHead(vmin, vmax, num_atoms)
        ]) for c in critic_network_heads]
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = snt.Sequential([
      networks.CriticMultiplexer(
          critic_network=critic_network_base),
      networks.Multihead(network_heads=critic_network_heads),
  ])
  return {
      'policy': policy_network,
      'critic': critic_network,
  }


def compute_action_norm(target_pi_samples: tf.Tensor,
                        target_q_target_pi_samples: tf.Tensor) -> tf.Tensor:
  """Compute Q-values for the action norm objective from action samples."""
  del target_q_target_pi_samples
  action_norm = tf.norm(target_pi_samples, ord=2, axis=-1)
  return tf.stop_gradient(-1 * action_norm)


def task_reward_fn(observation: tf.Tensor,
                   action: tf.Tensor,
                   reward: tf.Tensor) -> tf.Tensor:
  del observation, action
  return tf.stop_gradient(reward)


def make_objectives() -> Tuple[
    Sequence[mompo.RewardObjective], Sequence[mompo.QValueObjective]]:
  """Define the multiple objectives for the policy to learn."""
  task_reward = mompo.RewardObjective(
      name='task',
      reward_fn=task_reward_fn)
  action_norm = mompo.QValueObjective(
      name='action_norm_q',
      qvalue_fn=compute_action_norm)
  return [task_reward], [action_norm]


class MOMPOTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('distributional_critic', True),
      ('vanilla_critic', False))
  def test_mompo(self, distributional_critic):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(episode_length=10)
    spec = specs.make_environment_spec(environment)

    # Create objectives.
    reward_objectives, qvalue_objectives = make_objectives()
    num_critic_heads = len(reward_objectives)

    # Create networks.
    agent_networks = make_networks(
        spec.actions, num_critic_heads=num_critic_heads,
        distributional_critic=distributional_critic)

    # Construct the agent.
    agent = mompo.MultiObjectiveMPO(
        reward_objectives,
        qvalue_objectives,
        spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
