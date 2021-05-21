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

"""Shared helpers for different experiment flavours."""

from acme import wrappers
from dm_control import suite
import dm_env


def make_environment(evaluation: bool = False,
                     domain_name: str = 'cartpole',
                     task_name: str = 'balance') -> dm_env.Environment:
  """Implements a control suite environment factory."""
  # Nothing special to be done for evaluation environment.
  del evaluation

  environment = suite.load(domain_name, task_name)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment
