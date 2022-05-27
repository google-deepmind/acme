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

"""Example running a hyper sweep of DRQV2 + DMPO on the control suite."""

import functools
from typing import Dict, Sequence

from absl import app
from absl import flags
from acme import specs
from acme.agents.tf import dmpo
from acme.datasets import image_augmentation
import helpers
from acme.tf import networks
import launchpad as lp
import numpy as np
import sonnet as snt
import tensorflow as tf

# Flags which modify the behavior of the launcher.
FLAGS = flags.FLAGS
_MAX_ACTOR_STEPS = flags.DEFINE_integer(
    'max_actor_steps', None,
    'Number of actor steps to run; defaults to None for an endless loop.')
_DOMAIN = flags.DEFINE_string('domain', 'cartpole',
                              'Control suite domain name.')
_TASK = flags.DEFINE_string('task', 'balance', 'Control suite task name.')


def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (50, 1024, 1024),
    critic_layer_sizes: Sequence[int] = (50, 1024, 1024),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Dict[str, snt.Module]:
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      networks.LayerNormMLP(
          policy_layer_sizes,
          w_init=snt.initializers.Orthogonal(),
          activation=tf.nn.relu,
          activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          tanh_mean=False,
          init_scale=1.0,
          fixed_scale=False,
          use_tfd_independent=True,
          w_init=snt.initializers.Orthogonal())
  ])

  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = networks.CriticMultiplexer(
      observation_network=snt.Sequential([
          snt.Linear(critic_layer_sizes[0],
                     w_init=snt.initializers.Orthogonal()),
          snt.LayerNorm(
              axis=slice(1, None), create_scale=True, create_offset=True),
          tf.nn.tanh]),
      critic_network=snt.nets.MLP(
          critic_layer_sizes[1:],
          w_init=snt.initializers.Orthogonal(),
          activation=tf.nn.relu,
          activate_final=True),
      action_network=networks.ClipToSpec(action_spec))
  critic_network = snt.Sequential(
      [critic_network,
       networks.DiscreteValuedHead(vmin, vmax, num_atoms,
                                   w_init=snt.initializers.Orthogonal())
       ])
  observation_network = networks.DrQTorso()

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': observation_network,
  }


def main(_):
  # Configure the environment factory with requested task.
  make_environment = functools.partial(
      helpers.make_environment,
      domain_name=_DOMAIN.value,
      task_name=_TASK.value,
      from_pixels=True,
      frames_to_stack=3,
      flatten_stack=True,
      num_action_repeats=2)

  # Construct the program.
  program_builder = dmpo.DistributedDistributionalMPO(
      make_environment,
      make_networks,
      target_policy_update_period=100,
      max_actor_steps=_MAX_ACTOR_STEPS.value,
      num_actors=4,
      samples_per_insert=256,
      n_step=3,  # Reduce the n-step to account for action-repeat.
      observation_augmentation=image_augmentation.pad_and_crop,
  )

  # Launch experiment.
  lp.launch(programs=program_builder.build())


if __name__ == '__main__':
  app.run(main)
