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

"""Utilities for reverb-based adders."""

from typing import Dict, Mapping, Sequence

from acme import types
from acme.adders.reverb import base
from acme.utils import tf2_utils

import numpy as np
import tree


def final_step_like(step: base.Step,
                    next_observation: types.NestedArray) -> base.Step:
  """Return a list of steps with the final step zero-filled."""
  # Make zero-filled components so we can fill out the last step.
  zero_action, zero_reward, zero_discount, zero_extras = tree.map_structure(
      np.zeros_like, (step.action, step.reward, step.discount, step.extras))

  # Return a final step that only has next_observation.
  return base.Step(
      observation=next_observation,
      action=zero_action,
      reward=zero_reward,
      discount=zero_discount,
      extras=zero_extras)


def calculate_priorities(priority_fns: Mapping[str, base.PriorityFn],
                         steps: Sequence[base.Step]) -> Dict[str, float]:
  """Helper used to calculate the priority of a sequence of steps.

  This converts the sequence of steps into a PriorityFnInput tuple where the
  components of each step (actions, observations, etc.) are stacked along the
  time dimension.

  Priorities are calculated for the sequence or transition that starts from
  step[0].next_observation. As a result, the stack of observations comes from
  steps[0:] whereas all other components (e.g. actions, rewards, discounts,
  extras) corresponds to steps[1:].

  Note: this means that all components other than the observation will be
  ignored from step[0]. This also means that step[0] is allowed to correspond to
  an "initial step" in which case the action, reward, discount, and extras are
  each None, which is handled properly by this function.

  Args:
    priority_fns: a mapping from table names to priority functions (i.e. a
      callable of type PriorityFn). The given function will be used to generate
      the priority (a float) for the given table.
    steps: a list of Step objects used to compute the priorities.

  Returns:
    A dictionary mapping from table names to the priority (a float) for the
    given collection of steps.
  """

  # Stack the steps and wrap them as PrioityFnInput.
  fn_input = base.PriorityFnInput(*tf2_utils.stack_sequence_fields(steps))

  return {
      table: priority_fn(fn_input)
      for table, priority_fn in priority_fns.items()
  }
