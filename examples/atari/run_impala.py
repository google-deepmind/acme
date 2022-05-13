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

"""Run JAX IMPALA on Atari."""


from absl import app
from absl import flags
import acme
from acme.agents.jax import impala
import helpers

flags.DEFINE_string('level', 'PongNoFrameskip-v4', 'Which Atari level to play.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to train for.')

flags.DEFINE_integer('seed', 0, 'Random seed.')

FLAGS = flags.FLAGS


def main(_):
  env = helpers.make_environment(level=FLAGS.level, oar_wrapper=True)
  env_spec = acme.make_environment_spec(env)

  config = impala.IMPALAConfig(
      batch_size=16,
      sequence_period=10,
      seed=FLAGS.seed,
  )

  networks = impala.make_atari_networks(env_spec)
  agent = impala.IMPALAFromConfig(
      environment_spec=env_spec,
      forward_fn=networks.forward_fn,
      unroll_init_fn=networks.unroll_init_fn,
      unroll_fn=networks.unroll_fn,
      initial_state_fn=networks.initial_state_fn,
      config=config,
  )

  loop = acme.EnvironmentLoop(env, agent)
  loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
