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

"""Dataset related definitions and methods."""

import functools
import itertools
from typing import Iterator, Optional

from acme import types
from acme.jax import running_statistics
import jax
import jax.numpy as jnp
import rlds
import tensorflow as tf
import tree

# Keys in extras dictionary of the transitions.
# Total return over n-steps.
N_STEP_RETURN: str = 'n_step_return'
# Total return of the episode that the transition belongs to.
EPISODE_RETURN: str = 'episode_return'

# Indices of the time-batched transitions.
PREVIOUS: int = 0
CURRENT: int = 1
NEXT: int = 2


def _append_n_step_return(output, n_step_return):
  """Append n-step return to an output step."""
  output[N_STEP_RETURN] = n_step_return
  return output


def _append_episode_return(output, episode_return):
  """Append episode return to an output step."""
  output[EPISODE_RETURN] = episode_return
  return output


def _expand_scalars(output):
  """If rewards are scalar, expand them."""
  return tree.map_structure(tf.experimental.numpy.atleast_1d, output)


def episode_to_timestep_batch(
    episode: rlds.BatchedStep,
    return_horizon: int = 0,
    drop_return_horizon: bool = False,
    calculate_episode_return: bool = False) -> tf.data.Dataset:
  """Converts an episode into multi-timestep batches.

  Args:
    episode: Batched steps as provided directly by RLDS.
    return_horizon: int describing the horizon to which we should accumulate the
      return.
    drop_return_horizon: bool whether we should drop the last `return_horizon`
      steps to avoid mis-calculated returns near the end of the episode.
    calculate_episode_return: Whether to calculate episode return.  Can be an
      expensive operation on datasets with many episodes.

  Returns:
    rl_dataset.DatasetType of 3-batched transitions, with scalar rewards
      expanded to 1D rewards

  This means that for every step, the corresponding elements will be a batch of
  size 3, with the first batched element corresponding to *_t-1, the second to
  *_t and the third to *_t+1,  e.g. you can access the previous observation as:
  ```
  o_tm1 = el[types.OBSERVATION][0]
  ```
  Two additional keys can be added: 'R_t' which corresponds to the undiscounted
  return for horizon `return_horizon` from time t (always present), and
  'R_total' which corresponds to the total return of the associated episode (if
  `calculate_episode_return` is True). Rewards are converted to be (at least)
  one-dimensional, prior to batching (to avoid ()-shaped elements).

  In this example, 0-valued observations correspond to o_{t-1}, 1-valued
  observations correspond to o_t, and 2-valued observations correspond to
  s_{t+1}.  This same structure is true for all keys, except 'R_t' and 'R_total'
  which are both scalars.
  ```
  ipdb> el[types.OBSERVATION]
  <tf.Tensor: shape=(3, 11), dtype=float32, numpy=
  array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]], dtype=float32)>
  ```
  """
  steps = episode[rlds.STEPS]

  if drop_return_horizon:
    episode_length = steps.cardinality()
    steps = steps.take(episode_length - return_horizon)

  # Calculate n-step return:
  rewards = steps.map(lambda step: step[rlds.REWARD])
  batched_rewards = rlds.transformations.batch(
      rewards, size=return_horizon, shift=1, stride=1, drop_remainder=True)
  returns = batched_rewards.map(tf.math.reduce_sum)
  output = tf.data.Dataset.zip((steps, returns)).map(_append_n_step_return)

  # Calculate total episode return for potential filtering, use total # of steps
  # to calculate return.
  if calculate_episode_return:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    # Need to redefine this here to avoid a tf.data crash.
    rewards = steps.map(lambda step: step[rlds.REWARD])
    episode_return = rewards.reduce(dtype(0), lambda x, y: x + y)
    output = output.map(
        functools.partial(
            _append_episode_return, episode_return=episode_return))

  output = output.map(_expand_scalars)

  output = rlds.transformations.batch(
      output, size=3, shift=1, drop_remainder=True)
  return output


def _step_to_transition(rlds_step: rlds.BatchedStep) -> types.Transition:
  """Converts batched RLDS steps to batched transitions."""
  return types.Transition(
      observation=rlds_step[rlds.OBSERVATION],
      action=rlds_step[rlds.ACTION],
      reward=rlds_step[rlds.REWARD],
      discount=rlds_step[rlds.DISCOUNT],
      #  We provide next_observation if an algorithm needs it, however note that
      # it will only contain s_t and s_t+1, so will be one element short of all
      # other attributes (which contain s_t-1, s_t, s_t+1).
      next_observation=tree.map_structure(lambda x: x[1:],
                                          rlds_step[rlds.OBSERVATION]),
      extras={
          N_STEP_RETURN: rlds_step[N_STEP_RETURN],
      })


def episodes_to_timestep_batched_transitions(
    episode_dataset: tf.data.Dataset,
    return_horizon: int = 10,
    drop_return_horizon: bool = False,
    min_return_filter: Optional[float] = None) -> tf.data.Dataset:
  """Process an existing dataset converting it to episode to 3-transitions.

  A 3-transition is an Transition with each attribute having an extra dimension
  of size 3, representing 3 consecutive timesteps. Each 3-step object will be
  in random order relative to each other.  See `episode_to_timestep_batch` for
  more information.

  Args:
    episode_dataset: An RLDS dataset to process.
    return_horizon: The horizon we want calculate Monte-Carlo returns to.
    drop_return_horizon: Whether we should drop the last `return_horizon` steps.
    min_return_filter: Minimum episode return below which we drop an episode.

  Returns:
    A tf.data.Dataset of 3-transitions.
  """
  dataset = episode_dataset.interleave(
      functools.partial(
          episode_to_timestep_batch,
          return_horizon=return_horizon,
          drop_return_horizon=drop_return_horizon,
          calculate_episode_return=min_return_filter is not None),
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      deterministic=False)

  if min_return_filter is not None:

    def filter_on_return(step):
      return step[EPISODE_RETURN][0][0] > min_return_filter

    dataset = dataset.filter(filter_on_return)

  dataset = dataset.map(
      _step_to_transition, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return dataset


def get_normalization_stats(
    iterator: Iterator[types.Transition],
    num_normalization_batches: int = 50
) -> running_statistics.RunningStatisticsState:
  """Precomputes normalization statistics over a fixed number of batches.

  The iterator should contain batches of 3-transitions, i.e. with two leading
  dimensions, the first one denoting the batch dimension and the second one the
  previous, current and next timesteps. The statistics are calculated using the
  data of the previous timestep.

  Args:
    iterator: Iterator of batchs of 3-transitions.
    num_normalization_batches: Number of batches to calculate the statistics.

  Returns:
    RunningStatisticsState containing the normalization statistics.
  """
  # Set up normalization:
  example = next(iterator)
  unbatched_single_example = jax.tree_map(lambda x: x[0, PREVIOUS, :], example)
  mean_std = running_statistics.init_state(unbatched_single_example)

  for batch in itertools.islice(iterator, num_normalization_batches - 1):
    example = jax.tree_map(lambda x: x[:, PREVIOUS, :], batch)
    mean_std = running_statistics.update(mean_std, example)

  return mean_std
