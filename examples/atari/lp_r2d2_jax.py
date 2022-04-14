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

"""Example running R2D2, on Atari."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.jax import r2d2
from acme.agents.jax.r2d2 import networks as r2d2_networks
import helpers
import launchpad as lp

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'PongNoFrameskip-v4', 'Atari task name (str).')
flags.DEFINE_integer('num_actors', 4, 'Number of parallel actors.')


def main(_):
  # Access flag value.
  level = FLAGS.task
  environment_factory = (
      lambda seed: helpers.make_environment(level=level, oar_wrapper=True))
  config = r2d2.R2D2Config()
  def net_factory(spec: specs.EnvironmentSpec):
    return r2d2_networks.make_atari_networks(config.batch_size, env_spec=spec)

  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)

  program = r2d2.DistributedR2D2FromConfig(
      seed=0,
      environment_factory=environment_factory,
      network_factory=net_factory,
      config=config,
      num_actors=FLAGS.num_actors,
      environment_spec=env_spec,
  ).build()

  lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


if __name__ == '__main__':
  app.run(main)
