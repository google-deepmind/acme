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

"""Transition adders.

This implements an N-step transition adder which collapses trajectory sequences
into a single transition, simplifying to a simple transition adder when N=1.
"""

import copy
import itertools
import operator
from typing import Optional

from acme import specs
from acme import types
from acme.adders.reverb import base
from acme.adders.reverb import utils
from acme.utils import tree_utils

import numpy as np
import reverb
import tree


class NStepTransitionAdder(base.ReverbAdder):
  """An N-step transition adder.

  This will buffer a sequence of N timesteps in order to form a single N-step
  transition which is added to reverb for future retrieval.

  For N=1 the data added to replay will be a standard one-step transition which
  takes the form:

        (s_t, a_t, r_t, d_t, s_{t+1}, e_t)

  where:

    s_t = state observation at time t
    a_t = the action taken from s_t
    r_t = reward ensuing from action a_t
    d_t = environment discount ensuing from action a_t. This discount is
        applied to future rewards after r_t.
    e_t [Optional] = extra data that the agent persists in replay.

  For N greater than 1, transitions are of the form:

        (s_t, a_t, R_{t:t+n}, D_{t:t+n}, s_{t+N}, e_t),

  where:

    s_t = State (observation) at time t.
    a_t = Action taken from state s_t.
    g = the additional discount, used by the agent to discount future returns.
    R_{t:t+n} = N-step discounted return, i.e. accumulated over N rewards:
          R_{t:t+n} := r_t + g * d_t * r_{t+1} + ...
                           + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}.
    D_{t:t+n}: N-step product of agent discounts g_i and environment
      "discounts" d_i.
          D_{t:t+n} := g^{n-1} * d_{t} * ... * d_{t+n-1},
      For most environments d_i is 1 for all steps except the last,
      i.e. it is the episode termination signal.
    s_{t+n}: The "arrival" state, i.e. the state at time t+n.
    e_t [Optional]: A nested structure of any 'extras' the user wishes to add.

  Notes:
    - At the beginning and end of episodes, shorter transitions are added.
      That is, at the beginning of the episode, it will add:
            (s_0 -> s_1), (s_0 -> s_2), ..., (s_0 -> s_n), (s_1 -> s_{n+1})

      And at the end of the episode, it will add:
            (s_{T-n+1} -> s_T), (s_{T-n+2} -> s_T), ... (s_{T-1} -> s_T).
    - We add the *first* `extra` of each transition, not the *last*, i.e.
        if extras are provided, we get e_t, not e_{t+n}.
  """

  def __init__(
      self,
      client: reverb.Client,
      n_step: int,
      discount: float,
      priority_fns: Optional[base.PriorityFnMapping] = None,
  ):
    """Creates an N-step transition adder.

    Args:
      client: A `reverb.Client` to send the data to replay through.
      n_step: The "N" in N-step transition. See the class docstring for the
        precise definition of what an N-step transition is. `n_step` must be at
        least 1, in which case we use the standard one-step transition, i.e.
        (s_t, a_t, r_t, d_t, s_t+1, e_t).
      discount: Discount factor to apply. This corresponds to the
        agent's discount in the class docstring.
      priority_fns: See docstring for BaseAdder.

    Raises:
      ValueError: If n_step is less than 1.
    """
    # Makes the additional discount a float32, which means that it will be
    # upcast if rewards/discounts are float64 and left alone otherwise.
    self._discount = tree.map_structure(np.float32, discount)

    super().__init__(
        client=client,
        buffer_size=n_step,
        max_sequence_length=1,
        priority_fns=priority_fns)

  def _write(self):
    # NOTE: we do not check that the buffer is of length N here. This means
    # that at the beginning of an episode we will add the initial N-1
    # transitions (of size 1, 2, ...) and at the end of an episode (when
    # called from write_last) we will write the final transitions of size (N,
    # N-1, ...). See the Note in the docstring.

    # Form the n-step transition given the steps.
    observation = self._buffer[0].observation
    action = self._buffer[0].action
    extras = self._buffer[0].extras
    next_observation = self._next_observation

    # Give the same tree structure to the n-step return accumulator,
    # n-step discount accumulator, and self.discount, so that they can be
    # iterated in parallel using tree.map_structure.
    (n_step_return,
     total_discount,
     self_discount) = tree_utils.broadcast_structures(
         self._buffer[0].reward,
         self._buffer[0].discount,
         self._discount)

    # Copy total_discount, so that accumulating into it doesn't affect
    # _buffer[0].discount.
    total_discount = tree.map_structure(np.copy, total_discount)

    # Broadcast n_step_return to have the broadcasted shape of
    # reward * discount. Also copy, to avoid accumulating into
    # _buffer[0].reward.
    n_step_return = tree.map_structure(
        lambda r, d: np.copy(np.broadcast_to(r, np.broadcast(r, d).shape)),
        n_step_return,
        total_discount)

    # NOTE: total discount will have one less discount than it does
    # step.discounts. This is so that when the learner/update uses an additional
    # discount we don't apply it twice. Inside the following loop we will
    # apply this right before summing up the n_step_return.
    for step in itertools.islice(self._buffer, 1, None):
      (step_discount,
       step_reward,
       total_discount) = tree_utils.broadcast_structures(
           step.discount,
           step.reward,
           total_discount)

      # Equivalent to: `total_discount *= self._discount`.
      tree.map_structure(operator.imul, total_discount, self_discount)

      # Equivalent to: `n_step_return += step.reward * total_discount`.
      tree.map_structure(lambda nsr, sr, td: operator.iadd(nsr, sr * td),
                         n_step_return,
                         step_reward,
                         total_discount)

      # Equivalent to: `total_discount *= step.discount`.
      tree.map_structure(operator.imul, total_discount, step_discount)

    if extras:
      transition = (observation, action, n_step_return, total_discount,
                    next_observation, extras)
    else:
      transition = (observation, action, n_step_return, total_discount,
                    next_observation)

    # Create a list of steps.
    final_step = utils.final_step_like(self._buffer[0], next_observation)
    steps = list(self._buffer) + [final_step]

    # Calculate the priority for this transition.
    table_priorities = utils.calculate_priorities(self._priority_fns, steps)

    # Insert the transition into replay along with its priority.
    self._writer.append(transition)
    for table, priority in table_priorities.items():
      self._writer.create_item(
          table=table, num_timesteps=1, priority=priority)

  def _write_last(self):
    # Drain the buffer until there are no transitions.
    self._buffer.popleft()
    while self._buffer:
      self._write()
      self._buffer.popleft()

  @classmethod
  def signature(cls,
                environment_spec: specs.EnvironmentSpec,
                extras_spec: types.NestedSpec = ()):

    # This function currently assumes that self._discount is a scalar.
    # If it ever becomes a nested structure and/or a np.ndarray, this method
    # will need to know its structure / shape. This is because the signature
    # discount shape is the environment's discount shape and this adder's
    # discount shape broadcasted together. Also, the reward shape is this
    # signature discount shape broadcasted together with the environment
    # reward shape. As long as self._discount is a scalar, it will not affect
    # either the signature discount shape nor the signature reward shape, so we
    # can ignore it.

    rewards_spec, step_discounts_spec = tree_utils.broadcast_structures(
        environment_spec.rewards,
        environment_spec.discounts)
    rewards_spec = tree.map_structure(_broadcast_specs,
                                      rewards_spec,
                                      step_discounts_spec)
    step_discounts_spec = tree.map_structure(copy.deepcopy, step_discounts_spec)

    transition_spec = [
        environment_spec.observations,
        environment_spec.actions,
        rewards_spec,
        step_discounts_spec,
        environment_spec.observations,  # next_observation
    ]

    if extras_spec:
      transition_spec.append(extras_spec)

    return tree.map_structure_with_path(base.spec_like_to_tensor_spec,
                                        tuple(transition_spec))


def _broadcast_specs(*args: specs.Array) -> specs.Array:
  """Like np.broadcast, but for specs.Array.

  Args:
    *args: one or more specs.Array instances.

  Returns:
    A specs.Array with the broadcasted shape and dtype of the specs in *args.
  """
  bc_info = np.broadcast(*tuple(a.generate_value() for a in args))
  dtype = np.result_type(*tuple(a.dtype for a in args))
  return specs.Array(shape=bc_info.shape, dtype=dtype)

