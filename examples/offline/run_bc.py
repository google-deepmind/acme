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

import functools
import operator

from absl import app
from absl import flags
import acme
from acme import specs
from acme import types
from acme.agents.tf import actors
from acme.agents.tf.bc import learning
from acme.agents.tf.dqfd import bsuite_demonstrations
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import single_precision
import bsuite
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')

# Agent flags
flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_float('epsilon', 0., 'Epsilon for the epsilon greedy in the env.')
flags.DEFINE_integer('evaluate_every', 100, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')

FLAGS = flags.FLAGS


def make_policy_network(action_spec: specs.DiscreteArray) -> snt.Module:
  return snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([64, 64, action_spec.num_values]),
  ])


# TODO(b/152733199): Move this function to acme utils.
def _n_step_transition_from_episode(observations: types.NestedTensor,
                                    actions: tf.Tensor, rewards: tf.Tensor,
                                    discounts: tf.Tensor, n_step: int,
                                    additional_discount: float):
  """Produce Reverb-like N-step transition from a full episode.

  Observations, actions, rewards and discounts have the same length. This
  function will ignore the first reward and discount and the last action.

  Args:
    observations: [L, ...] Tensor.
    actions: [L, ...] Tensor.
    rewards: [L] Tensor.
    discounts: [L] Tensor.
    n_step: number of steps to squash into a single transition.
    additional_discount: discount to use for TD updates.

  Returns:
    (o_t, a_t, r_t, d_t, o_tp1) tuple.
  """

  max_index = tf.shape(rewards)[0] - 1
  first = tf.random.uniform(
      shape=(), minval=0, maxval=max_index - 1, dtype=tf.int32)
  last = tf.minimum(first + n_step, max_index)

  o_t = tree.map_structure(operator.itemgetter(first), observations)
  a_t = tree.map_structure(operator.itemgetter(first), actions)
  o_tp1 = tree.map_structure(operator.itemgetter(last), observations)

  # 0, 1, ..., n-1.
  discount_range = tf.cast(tf.range(last - first), tf.float32)
  # 1, g, ..., g^{n-1}.
  additional_discounts = tf.pow(additional_discount, discount_range)
  # 1, d_t, d_t * d_{t+1}, ..., d_t * ... * d_{t+n-2}.
  discounts = tf.concat([[1.], tf.math.cumprod(discounts[first:last - 1])], 0)
  # 1, g * d_t, ..., g^{n-1} * d_t * ... * d_{t+n-2}.
  discounts *= additional_discounts
  # r_t + g * d_t * r_{t+1} + ... + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}
  # We have to shift rewards by one so last=max_index corresponds to transitions
  # that include the last reward.
  r_t = tf.reduce_sum(rewards[first + 1:last + 1] * discounts)

  # g^{n-1} * d_{t} * ... * d_{t+n-1}.
  d_t = discounts[-1]

  # Reverb requires every sample to be given a key and priority.
  # In the supervised learning case for BC, neither of those will be used.
  # We set the key to `0` and the priorities probabilities to `1`, but that
  # should not matter much.
  key = tf.constant(0, tf.uint64)
  probability = tf.constant(1.0, tf.float64)
  table_size = tf.constant(1, tf.int64)
  priority = tf.constant(1.0, tf.float64)
  times_sampled = tf.constant(1, tf.int32)
  info = reverb.SampleInfo(
      key=key,
      probability=probability,
      table_size=table_size,
      priority=priority,
      times_sampled=times_sampled,
  )

  return reverb.ReplaySample(info=info, data=(o_t, a_t, r_t, d_t, o_tp1))


def main(_):
  # Create an environment and grab the spec.
  raw_environment = bsuite.load_and_record_to_csv(
      bsuite_id=FLAGS.bsuite_id,
      results_dir=FLAGS.results_dir,
      overwrite=FLAGS.overwrite,
  )
  environment = single_precision.SinglePrecisionWrapper(raw_environment)
  environment_spec = specs.make_environment_spec(environment)

  # Build demonstration dataset.
  if hasattr(raw_environment, 'raw_env'):
    raw_environment = raw_environment.raw_env

  batch_dataset = bsuite_demonstrations.make_dataset(raw_environment,
                                                     stochastic=False)
  # Combine with demonstration dataset.
  transition = functools.partial(
      _n_step_transition_from_episode, n_step=1, additional_discount=1.)

  dataset = batch_dataset.map(transition)

  # Batch and prefetch.
  dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  # Create the networks to optimize.
  policy_network = make_policy_network(environment_spec.actions)

  # If the agent is non-autoregressive use epsilon=0 which will be a greedy
  # policy.
  evaluator_network = snt.Sequential([
      policy_network,
      lambda q: trfl.epsilon_greedy(q, epsilon=FLAGS.epsilon).sample(),
  ])

  # Ensure that we create the variables before proceeding (maybe not needed).
  tf2_utils.create_variables(policy_network, [environment_spec.observations])

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
  learner = learning.BCLearner(
      network=policy_network,
      learning_rate=FLAGS.learning_rate,
      dataset=dataset,
      counter=learner_counter)

  # Run the environment loop.
  while True:
    for _ in range(FLAGS.evaluate_every):
      learner.step()
    learner_counter.increment(learner_steps=FLAGS.evaluate_every)
    eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
