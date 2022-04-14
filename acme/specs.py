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

"""Objects which specify the input/output spaces of an environment.

This module exposes the same spec classes as `dm_env` as well as providing an
additional `EnvironmentSpec` class which collects all of the specs for a given
environment. An `EnvironmentSpec` instance can be created directly or by using
the `make_environment_spec` helper given a `dm_env.Environment` instance.
"""

from typing import Any, NamedTuple

import dm_env
from dm_env import specs

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


class EnvironmentSpec(NamedTuple):
  """Full specification of the domains used by a given environment."""
  # TODO(b/144758674): Use NestedSpec type here.
  observations: Any
  actions: Any
  rewards: Any
  discounts: Any


def make_environment_spec(environment: dm_env.Environment) -> EnvironmentSpec:
  """Returns an `EnvironmentSpec` describing values used by an environment."""
  return EnvironmentSpec(
      observations=environment.observation_spec(),
      actions=environment.action_spec(),
      rewards=environment.reward_spec(),
      discounts=environment.discount_spec())
