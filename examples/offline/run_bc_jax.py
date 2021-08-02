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

"""An example BC running on BSuite."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.jax import actors
from acme.agents.jax import bc
from acme.examples.offline import bc_utils
from acme.jax import learning
from acme.jax import variable_utils
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
from jax.scipy import special
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
  network = bc_utils.make_network(environment_spec)

  key = jax.random.PRNGKey(FLAGS.seed)
  key, key1 = jax.random.split(key, 2)

  def logp_fn(logits, actions):
    logits_actions = jnp.sum(
        jax.nn.one_hot(actions, logits.shape[-1]) * logits, axis=-1)
    logits_actions = logits_actions - special.logsumexp(logits, axis=-1)
    return logits_actions

  loss_fn = bc.logp(logp_fn=logp_fn)

  # The learner core is a double of functions to initialize the training state
  # (init) and update it given a batch of data (step)
  learner_core = bc.make_bc_learner_core(
      network=network,
      loss_fn=loss_fn,
      optimizer=optax.adam(FLAGS.learning_rate))

  learner = learning.DefaultJaxLearner(learner_core, dataset, key1)

  def evaluator_network(params: hk.Params, key: jnp.DeviceArray,
                        observation: jnp.DeviceArray) -> jnp.DeviceArray:
    dist_params = network.apply(params, observation)
    return rlax.epsilon_greedy(FLAGS.evaluation_epsilon).sample(
        key, dist_params)

  evaluator = actors.FeedForwardActor(
      policy=evaluator_network,
      random_key=key,
      # Inference happens on CPU, so it's better to move variables there too.
      variable_client=variable_utils.VariableClient(
          learner, 'policy', device='cpu'))

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
