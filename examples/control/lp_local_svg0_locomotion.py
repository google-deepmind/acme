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

"""Example running SVG0 on the locomotion predicate tasks.

It runs the distributed agent using Launchpad runtime specified by
--lp_launch_type flag.
"""

from absl import app
from absl import flags

from acme import wrappers
from acme.agents.tf import svg0_prior
from acme.utils import lp_utils
import dm_env
import launchpad as lp

from deepmind_research.box_arrangement import task_examples # type: ignore


FLAGS = flags.FLAGS


def make_gtt_environment(evaluation: bool = False,) -> dm_env.Environment:
  """Implements a predicate go to target environment factory."""
  # Nothing special to be done for evaluation environment.
  del evaluation

  environment = task_examples.go_to_k_targets()
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment


def main(_):
  environment_factory = lp_utils.partial_kwargs(make_gtt_environment)

  batch_size = 512
  sequence_length = 10
  num_actors = 32
  # Policy sees full task information.
  policy_keys = (
      # Proprio
      "walker/body_height",
      "walker/end_effectors_pos",
      "walker/joints_pos",
      "walker/joints_vel",
      "walker/sensors_accelerometer",
      "walker/sensors_gyro",
      "walker/sensors_velocimeter",
      "walker/world_zaxis",
      "walker/target_positions",
      "predicate_0")
  # Prior sees only proprioception.
  prior_keys = (
      # Proprio
      "walker/body_height",
      "walker/end_effectors_pos",
      "walker/joints_pos",
      "walker/joints_vel",
      "walker/sensors_accelerometer",
      "walker/sensors_gyro",
      "walker/sensors_velocimeter",
      "walker/world_zaxis",
  )

  program = svg0_prior.DistributedSVG0(
      environment_factory=environment_factory,
      network_factory=lp_utils.partial_kwargs(
          svg0_prior.make_network_with_prior,
          policy_keys=policy_keys,
          prior_keys=prior_keys,
      ),
      batch_size=batch_size,
      sequence_length=sequence_length,
      samples_per_insert=None,
      entropy_regularizer_cost=1e-3,
      distillation_cost=0.0,
      max_replay_size=int(1e6),
      num_actors=num_actors).build()

  # Launch experiment.
  lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))


if __name__ == "__main__":
  app.run(main)
