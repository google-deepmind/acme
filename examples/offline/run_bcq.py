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

r"""Run BCQ offline agent on Atari RL Unplugged datasets.

Instructions:

1 - Download dataset:
> gsutil cp gs://rl_unplugged/atari/Pong/run_1-00000-of-00100 \
    /tmp/dataset/Pong/run_1-00000-of-00001

2 - Install RL Unplugged dependencies:
https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged#running-the-code

3 - Download RL Unplugged library:
> git clone https://github.com/deepmind/deepmind-research.git deepmind_research

4 - Run script:
> python -m run_atari_bcq --dataset_path=/tmp/dataset --game=Pong --run=1 \
    --num_shards=1
"""

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.tf import actors
from acme.agents.tf import bcq
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import sonnet as snt
import tensorflow as tf
import trfl

from deepmind_research.rl_unplugged import atari # type: ignore

# Atari dataset flags
flags.DEFINE_string('dataset_path', None, 'Dataset path.')
flags.DEFINE_string('game', 'Pong', 'Dataset path.')
flags.DEFINE_integer('run', 1, 'Dataset path.')
flags.DEFINE_integer('num_shards', 100, 'Number of dataset shards.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')

# Agent flags
flags.DEFINE_float('bcq_threshold', 0.5, 'BCQ threshold.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount.')
flags.DEFINE_float('importance_sampling_exponent', 0.2,
                   'Importance sampling exponent.')
flags.DEFINE_integer('target_update_period', 2500,
                     ('Number of learner steps to perform before updating'
                      'the target networks.'))

# Evaluation flags.
flags.DEFINE_float('epsilon', 0., 'Epsilon for the epsilon greedy in the env.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')

FLAGS = flags.FLAGS


def make_network(action_spec: specs.DiscreteArray) -> snt.Module:
  return snt.Sequential([
      lambda x: tf.image.convert_image_dtype(x, tf.float32),
      networks.DQNAtariNetwork(action_spec.num_values)
  ])


def main(_):
  # Create an environment and grab the spec.
  environment = atari.environment(FLAGS.game)
  environment_spec = specs.make_environment_spec(environment)

  # Create dataset.
  dataset = atari.dataset(path=FLAGS.dataset_path,
                          game=FLAGS.game,
                          run=FLAGS.run,
                          num_shards=FLAGS.num_shards)
  # Discard extra inputs
  dataset = dataset.map(lambda x: x._replace(data=x.data[:5]))

  # Batch and prefetch.
  dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  # Build network.
  g_network = make_network(environment_spec.actions)
  q_network = make_network(environment_spec.actions)
  network = networks.DiscreteFilteredQNetwork(g_network=g_network,
                                              q_network=q_network,
                                              threshold=FLAGS.bcq_threshold)
  tf2_utils.create_variables(network, [environment_spec.observations])

  evaluator_network = snt.Sequential([
      q_network,
      lambda q: trfl.epsilon_greedy(q, epsilon=FLAGS.epsilon).sample(),
  ])

  # Counters.
  counter = counting.Counter()
  learner_counter = counting.Counter(counter, prefix='learner')

  # Create the actor which defines how we take actions.
  evaluation_network = actors.FeedForwardActor(evaluator_network)

  eval_loop = acme.EnvironmentLoop(
      environment=environment,
      actor=evaluation_network,
      counter=counter,
      logger=loggers.TerminalLogger('evaluation', time_delta=1.))

  # The learner updates the parameters (and initializes them).
  learner = bcq.DiscreteBCQLearner(
      network=network,
      dataset=dataset,
      learning_rate=FLAGS.learning_rate,
      discount=FLAGS.discount,
      importance_sampling_exponent=FLAGS.importance_sampling_exponent,
      target_update_period=FLAGS.target_update_period,
      counter=counter)

  # Run the environment loop.
  while True:
    for _ in range(FLAGS.evaluate_every):
      learner.step()
    learner_counter.increment(learner_steps=FLAGS.evaluate_every)
    eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
