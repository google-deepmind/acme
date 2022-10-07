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

"""ActorCore wrapper to use observation stacking."""

from typing import Any, Mapping, NamedTuple, Tuple

from acme import specs
from acme import types as acme_types
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils as jax_utils
from acme.tf import utils as tf_utils
import jax
import jax.numpy as jnp
import reverb
import tensorflow as tf
import tree

ActorState = Any
Observation = networks_lib.Observation
Action = networks_lib.Action
Params = networks_lib.Params


class StackerState(NamedTuple):
  stack: jax.Array  # Observations stacked along the final dimension.
  needs_reset: jax.Array  # A scalar boolean.


class StackingActorState(NamedTuple):
  actor_state: ActorState
  stacker_state: StackerState


# TODO(bshahr): Consider moving to jax_utils, extending current tiling function.
def tile_nested_array(nest: acme_types.NestedArray, num: int, axis: int):

  def _tile_array(array: jnp.ndarray) -> jnp.ndarray:
    reps = [1] * array.ndim
    reps[axis] = num
    return jnp.tile(array, reps)

  return jax.tree_map(_tile_array, nest)


class ObservationStacker:
  """Class used to handle agent-side observation stacking.

  Once an ObservationStacker is initialized and an initial_state is obtained
  from it, one can stack nested observations by simply calling the
  ObservationStacker and passing it the new observation and current state of its
  observation stack.

  See also observation_stacking.wrap_actor_core for hints on how to use it.
  """

  def __init__(self,
               observation_spec: acme_types.NestedSpec,
               stack_size: int = 4):

    def _repeat_observation(state: StackerState,
                            first_observation: Observation) -> StackerState:
      return state._replace(
          needs_reset=jnp.array(False),
          stack=tile_nested_array(first_observation, stack_size - 1, axis=-1))

    self._zero_stack = tile_nested_array(
        jax_utils.zeros_like(observation_spec), stack_size - 1, axis=-1)
    self._repeat_observation = _repeat_observation

  def __call__(self, inputs: Observation,
               state: StackerState) -> Tuple[Observation, StackerState]:

    # If this is a first observation, initialize the stack by repeating it,
    # otherwise leave it intact.
    state = jax.lax.cond(
        state.needs_reset,
        self._repeat_observation,
        lambda state, *args: state,  # No-op on state.
        state,
        inputs)

    # Concatenate frames along the final axis (assumed to be for channels).
    output = jax.tree_map(lambda *x: jnp.concatenate(x, axis=-1),
                          state.stack, inputs)

    # Update the frame stack by adding the input and dropping the first
    # observation in the stack. Note that we use the final dimension as each
    # leaf in the nested observation may have a different last dim.
    new_state = state._replace(
        stack=jax.tree_map(lambda x, y: y[..., x.shape[-1]:], inputs, output))

    return output, new_state

  def initial_state(self) -> StackerState:
    return StackerState(stack=self._zero_stack, needs_reset=jnp.array(True))


def get_adjusted_environment_spec(environment_spec: specs.EnvironmentSpec,
                                  stack_size: int) -> specs.EnvironmentSpec:
  """Returns a spec where the observation spec accounts for stacking."""

  def stack_observation_spec(obs_spec: specs.Array) -> specs.Array:
    """Adjusts last axis shape to account for observation stacking."""
    new_shape = obs_spec.shape[:-1] + (obs_spec.shape[-1] * stack_size,)
    return obs_spec.replace(shape=new_shape)

  adjusted_observation_spec = jax.tree_map(stack_observation_spec,
                                           environment_spec.observations)

  return environment_spec._replace(observations=adjusted_observation_spec)


def wrap_actor_core(
    actor_core: actor_core_lib.ActorCore,
    observation_spec: specs.Array,
    num_stacked_observations: int = 1) -> actor_core_lib.ActorCore:
  """Wraps an actor core so that it performs observation stacking."""

  if num_stacked_observations <= 0:
    raise ValueError(
        'Number of stacked observations must be strictly positive.'
        f' Received num_stacked_observations={num_stacked_observations}.')

  if num_stacked_observations == 1:
    # Return unwrapped core when a trivial stack size is requested.
    return actor_core

  obs_stacker = ObservationStacker(
      observation_spec=observation_spec, stack_size=num_stacked_observations)

  def init(key: jax_types.PRNGKey) -> StackingActorState:
    return StackingActorState(
        actor_state=actor_core.init(key),
        stacker_state=obs_stacker.initial_state())

  def select_action(
      params: Params,
      observations: Observation,
      state: StackingActorState,
  ) -> Tuple[Action, StackingActorState]:

    stacked_observations, stacker_state = obs_stacker(observations,
                                                      state.stacker_state)

    actions, actor_state = actor_core.select_action(params,
                                                    stacked_observations,
                                                    state.actor_state)
    new_state = StackingActorState(
        actor_state=actor_state, stacker_state=stacker_state)

    return actions, new_state

  def get_extras(state: StackingActorState) -> Mapping[str, jnp.ndarray]:
    return actor_core.get_extras(state.actor_state)

  return actor_core_lib.ActorCore(
      init=init, select_action=select_action, get_extras=get_extras)


def stack_reverb_observation(sample: reverb.ReplaySample,
                             stack_size: int) -> reverb.ReplaySample:
  """Stacks observations in a Reverb sample.

  This function is meant to be used on the dataset creation side as a
  post-processing function before batching.

  Warnings!
    * Only works if SequenceAdder is in end_of_episode_behavior=CONTINUE mode.
    * Only tested on RGB and scalar (shape = (1,)) observations.
    * At episode starts, this function repeats the first observation to form a
      stack. Could consider using zeroed observations instead.
    * At episode starts, this function always selects the latest possible
      stacked trajectory. Could consider randomizing the start index of the
      sequence.

  Args:
    sample: A sample coming from a Reverb replay table. Should be an unbatched
      sequence so that sample.data.observation is a nested structure of
      time-major tensors.
    stack_size: Number of observations to stack.

  Returns:
    A new sample where sample.data.observation has the same nested structure as
    the incoming sample but with every tensor having its final dimension
    multiplied by `stack_size`.
  """

  def _repeat_first(sequence: tf.Tensor) -> tf.Tensor:
    repeated_first_step = tf_utils.tile_tensor(sequence[0], stack_size - 1)
    return tf.concat([repeated_first_step, sequence], 0)[:-(stack_size - 1)]

  def _stack_observation(observation: tf.Tensor) -> tf.Tensor:
    stack = [tf.roll(observation, i, axis=0) for i in range(stack_size)]
    stack.reverse()  # Reverse stack order to be chronological.
    return tf.concat(stack, axis=-1)

  # Maybe repeat the first observation, if at the start of an episode.
  data = tf.cond(sample.data.start_of_episode[0],
                 lambda: tree.map_structure(_repeat_first, sample.data),
                 lambda: sample.data)

  # Stack observation in the sample's data.
  data_with_stacked_obs = data._replace(
      observation=tree.map_structure(_stack_observation, data.observation))

  # Truncate the start of the sequence due to the first stacks containing the
  # final observations that were rolled over to the start.
  data = tree.map_structure(lambda x: x[stack_size - 1:], data_with_stacked_obs)

  return reverb.ReplaySample(info=sample.info, data=data)
