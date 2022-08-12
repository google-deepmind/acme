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

"""Fake (mock) components for multiagent testing."""

from typing import Dict, List

from acme import specs
import numpy as np


def _make_multiagent_spec(agent_indices: List[str]) -> Dict[str, specs.Array]:
  """Returns dummy multiagent sub-spec (e.g., observation or action spec).

  Args:
    agent_indices: a list of agent indices.
  """
  return {
      agent_id: specs.BoundedArray((1,), np.float32, 0, 1)
      for agent_id in agent_indices
  }


def make_multiagent_environment_spec(
    agent_indices: List[str]) -> specs.EnvironmentSpec:
  """Returns dummy multiagent environment spec.

  Args:
    agent_indices: a list of agent indices.
  """
  action_spec = _make_multiagent_spec(agent_indices)
  discount_spec = specs.BoundedArray((), np.float32, 0.0, 1.0)
  observation_spec = _make_multiagent_spec(agent_indices)
  reward_spec = _make_multiagent_spec(agent_indices)
  return specs.EnvironmentSpec(
      actions=action_spec,
      discounts=discount_spec,
      observations=observation_spec,
      rewards=reward_spec)
