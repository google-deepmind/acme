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

"""Helpers functions for imitation tasks."""
from typing import Tuple

from acme import wrappers

import dm_env
import gym
import numpy as np
import tensorflow as tf


DATASET_NAMES = {
    'HalfCheetah-v2': 'locomotion/halfcheetah_sac_1M_single_policy_stochastic',
    'Ant-v2': 'locomotion/ant_sac_1M_single_policy_stochastic',
    'Walker2d-v2': 'locomotion/walker2d_sac_1M_single_policy_stochastic',
    'Hopper-v2': 'locomotion/hopper_sac_1M_single_policy_stochastic',
    'Humanoid-v2': 'locomotion/humanoid_sac_15M_single_policy_stochastic'
}


def get_dataset_name(env_name: str) -> str:
  return DATASET_NAMES[env_name]


def get_observation_stats(transitions_iterator: tf.data.Dataset
                          ) -> Tuple[np.float64, np.float64]:
  """Returns scale and shift of the observations in demonstrations."""
  observations = [step.observation for step in transitions_iterator]
  mean = np.mean(observations, axis=0, dtype='float64')
  std = np.std(observations, axis=0, dtype='float64')

  shift = - mean
  # The std is set to 1 if the observation values are below a threshold.
  # This prevents normalizing observation values that are constant (which can
  # be problematic with e.g. demonstrations coming from a different version
  # of the environment and where the constant values are slightly different).
  scale = 1 / ((std < 1e-6) + std)
  return shift, scale


def make_environment(
    task: str = 'MountainCarContinuous-v0') -> dm_env.Environment:
  """Creates an OpenAI Gym environment."""

  # Load the gym environment.
  environment = gym.make(task)

  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  # Clip the action returned by the agent to the environment spec.
  environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment
