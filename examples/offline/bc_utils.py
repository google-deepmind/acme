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

"""Utils for running behavioral cloning.
"""
import functools
import operator
from typing import Callable

from acme import core
from acme import environment_loop
from acme import specs
from acme import types
from acme.agents.jax import actors
from acme.agents.tf.dqfd import bsuite_demonstrations
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import single_precision
import bsuite
import dm_env
import haiku as hk
import tensorflow as tf
import tree


def make_network(
    spec: specs.EnvironmentSpec) -> networks_lib.FeedForwardNetwork:
  """Creates networks used by the agent."""
  num_actions = spec.actions.num_values

  def actor_fn(obs, is_training=True, key=None):
    # is_training and key allows to utilize train/test dependant modules
    # like dropout.
    del is_training
    del key
    mlp = hk.Sequential(
        [hk.Flatten(),
         hk.nets.MLP([64, 64, num_actions])])
    return mlp(obs)

  policy = hk.without_apply_rng(hk.transform(actor_fn, apply_rng=True))

  # Create dummy observations to create network parameters.
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  network = networks_lib.FeedForwardNetwork(
      lambda key: policy.init(key, dummy_obs), policy.apply)

  return network


def _n_step_transition_from_episode(
    observations: types.NestedTensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    discounts: tf.Tensor, n_step: int,
    additional_discount: float) -> types.Transition:
  """Produce Reverb-like N-step transition from a full episode.

  Observations, actions, rewards and discounts have the same length. This
  function will ignore the first reward and discount and the last action.

  Args:
    observations: [episode_length, ...] Tensor.
    actions: [episode_length, ...] Tensor.
    rewards: [episode_length] Tensor.
    discounts: [episode_length] Tensor.
    n_step: number of steps to squash into a single transition.
    additional_discount: discount to use for TD updates.

  Returns:
    A types.Transition.
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

  return types.Transition(o_t, a_t, r_t, d_t, o_tp1)


def make_environment(training: bool = True):
  del training
  env = bsuite.load(experiment_name='deep_sea', kwargs={'size': 10})
  return single_precision.SinglePrecisionWrapper(env)


def make_demonstrations(env: dm_env.Environment,
                        batch_size: int) -> tf.data.Dataset:
  """Prepare the dataset of demonstrations."""
  batch_dataset = bsuite_demonstrations.make_dataset(env, stochastic=False)
  # Combine with demonstration dataset.
  transition = functools.partial(
      _n_step_transition_from_episode, n_step=1, additional_discount=1.)

  dataset = batch_dataset.map(transition)

  # Batch and prefetch.
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset


def make_actor_evaluator(
    environment_factory: Callable[[bool], dm_env.Environment],
    evaluator_network: actors.FeedForwardPolicy,
) -> jax_types.EvaluatorFactory:
  """Makes an evaluator that runs the agent on the environment.

  Args:
    environment_factory: Function that creates a dm_env.
    evaluator_network: Network to be use by the actor.

  Returns:
    actor_evaluator: Function that returns a Worker that will be executed
      by launchpad.
  """
  def actor_evaluator(
      random_key: networks_lib.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
  ):
    """The evaluation process."""
    # Create the actor loading the weights from variable source.
    actor = actors.FeedForwardActor(
        policy=evaluator_network,
        random_key=random_key,
        # Inference happens on CPU, so it's better to move variables there too.
        variable_client=variable_utils.VariableClient(
            variable_source, 'policy', device='cpu'))

    # Logger.
    logger = loggers.make_default_logger(
        'evaluator', steps_key='evaluator_steps')

    # Create environment and evaluator networks
    environment = environment_factory(False)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(
        environment,
        actor,
        counter,
        logger,
    )

  return actor_evaluator
