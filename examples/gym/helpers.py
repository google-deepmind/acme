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

"""OpenAI Gym environment factory."""

from typing import Callable, Mapping, Sequence

from absl import flags
from acme import specs
from acme import types
from acme import wrappers
from acme.datasets import tfds
from acme.jax import utils
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from acme.utils.loggers import tf_summary
import dm_env
import gym
import jax
import numpy as np
import sonnet as snt

FLAGS = flags.FLAGS
# We want all examples to have this same flag defined.
flags.DEFINE_string('tfsummary_logdir', '',
                    'Root directory for logging tf.summary.')

TASKS = {
    'debug': ['MountainCarContinuous-v0'],
    'default': [
        'HalfCheetah-v2', 'Hopper-v2', 'InvertedDoublePendulum-v2',
        'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2'
    ],
}


def make_environment(
    evaluation: bool = False,
    task: str = 'MountainCarContinuous-v0') -> dm_env.Environment:
  """Creates an OpenAI Gym environment."""
  del evaluation

  # Load the gym environment.
  environment = gym.make(task)

  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  # Clip the action returned by the agent to the environment spec.
  environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment


def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
  """Creates networks used by the agent."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(action_spec.shape, dtype=int)

  # Create the shared observation network; here simply a state-less operation.
  observation_network = tf2_utils.batch_concat

  # Create the policy network.
  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.NearZeroInitializedLinear(num_dimensions),
      networks.TanhToSpec(action_spec),
  ])

  # Create the critic network.
  critic_network = snt.Sequential([
      # The multiplexer concatenates the observations/actions.
      networks.CriticMultiplexer(),
      networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
      networks.DiscreteValuedHead(vmin, vmax, num_atoms),
  ])

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': observation_network,
  }


def make_demonstration_iterator(batch_size: int,
                                dataset_name: str,
                                seed: int = 0):
  dataset = tfds.get_tfds_dataset(dataset_name)
  return tfds.JaxInMemoryRandomSampleIterator(dataset, jax.random.PRNGKey(seed),
                                              batch_size)


# TODO(sinopalnikov): make it shareable across all examples, not only Gym ones.
def create_logger_fn() -> Callable[[], loggers.Logger]:
  """Returns a function that creates logger instances."""
  if not FLAGS.tfsummary_logdir:
    # Use default logger.
    return lambda: None

  def create_logger() -> loggers.Logger:
    label = 'learner'
    default_learner_logger = loggers.make_default_logger(
        label=label,
        save_data=False,
        time_delta=10.0,
        asynchronous=True,
        steps_key='learner_steps')
    tf_summary_logger = tf_summary.TFSummaryLogger(
        logdir=FLAGS.tfsummary_logdir, label=label)

    # Sending logs to each of these targets.
    destinations = [default_learner_logger, tf_summary_logger]
    logger = loggers.aggregators.Dispatcher(
        destinations, serialize_fn=utils.fetch_devicearray)
    return logger

  return create_logger
