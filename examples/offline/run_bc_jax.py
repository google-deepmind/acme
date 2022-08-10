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

"""An example BC running on BSuite."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import bc
from acme.examples.offline import bc_utils
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax

# Agent flags
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('evaluation_epsilon', 0.,
                   'Epsilon for the epsilon greedy in the evaluation agent.')
flags.DEFINE_integer('evaluate_every', 20, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer('seed', 0, 'Random seed for learner and evaluator.')

FLAGS = flags.FLAGS


def main(_):
  # Create an environment and grab the spec.
  environment = bc_utils.make_environment()
  environment_spec = specs.make_environment_spec(environment)

  # Unwrap the environment to get the demonstrations.
  dataset = bc_utils.make_demonstrations(environment.environment,
                                         FLAGS.batch_size)
  dataset = dataset.as_numpy_iterator()

  # Create the networks to optimize.
  bc_networks = bc_utils.make_network(environment_spec)

  key = jax.random.PRNGKey(FLAGS.seed)
  key, key1 = jax.random.split(key, 2)

  loss_fn = bc.logp()

  learner = bc.BCLearner(
      networks=bc_networks,
      random_key=key1,
      loss_fn=loss_fn,
      optimizer=optax.adam(FLAGS.learning_rate),
      prefetching_iterator=utils.sharded_prefetch(dataset),
      num_sgd_steps_per_step=1)

  def evaluator_network(params: hk.Params, key: jnp.DeviceArray,
                        observation: jnp.DeviceArray) -> jnp.DeviceArray:
    dist_params = bc_networks.policy_network.apply(params, observation)
    return rlax.epsilon_greedy(FLAGS.evaluation_epsilon).sample(
        key, dist_params)

  actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
      evaluator_network)
  variable_client = variable_utils.VariableClient(
      learner, 'policy', device='cpu')
  evaluator = actors.GenericActor(
      actor_core, key, variable_client, backend='cpu')

  eval_loop = acme.EnvironmentLoop(
      environment=environment,
      actor=evaluator,
      logger=loggers.TerminalLogger('evaluation', time_delta=0.))

  # Run the environment loop.
  while True:
    for _ in range(FLAGS.evaluate_every):
      learner.step()
    eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
