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

"""Launch MPO agent on the control suite via Launchpad."""

import functools
from typing import Dict, Sequence

from absl import app
from absl import flags
from acme import specs
from acme import types
from acme.agents.tf import mpo
import helpers
from acme.tf import networks
from acme.tf import utils as tf2_utils
import launchpad as lp
import numpy as np
import sonnet as snt


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
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
) -> Dict[str, types.TensorTransformation]:
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions, init_scale=0.7, use_tfd_independent=True)
  ])

  # The multiplexer concatenates the (maybe transformed) observations/actions.
  multiplexer = networks.CriticMultiplexer(
      action_network=networks.ClipToSpec(action_spec))
  critic_network = snt.Sequential([
      multiplexer,
      networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
      networks.NearZeroInitializedLinear(1),
  ])

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': tf2_utils.batch_concat,
  }


def main(_):
  # Configure the environment factory with requested task.
  make_environment = functools.partial(
      helpers.make_environment,
      domain_name=_DOMAIN.value,
      task_name=_TASK.value)

  # Construct the program.
  program_builder = mpo.DistributedMPO(
      make_environment,
      make_networks,
      target_policy_update_period=25,
      max_actor_steps=_MAX_ACTOR_STEPS.value,
      num_actors=4)

  lp.launch(programs=program_builder.build())


if __name__ == '__main__':
  app.run(main)
