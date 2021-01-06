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

from acme import wrappers
import dm_env
import gym

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
