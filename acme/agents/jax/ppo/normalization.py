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

"""Utilities for normalization."""
import dataclasses
from typing import Any, Callable, NamedTuple, Optional

from acme import core
from acme import types
from acme.jax import running_statistics
from acme.jax import utils
from acme.jax import variable_utils
import dm_env
import jax
import jax.numpy as jnp

NormalizationParams = Any
RunningStatisticsState = running_statistics.RunningStatisticsState


@dataclasses.dataclass
class NormalizationFns:
  """Holds pure functions for normalization.

  Attributes:
    init: A pure function: ``params = init()``
    normalize: A pure function: ``norm_x = normalize(x, params)``
    update: A pure function: ``params = update(params, x, pmap_axis_name)``
  """
  # Returns the initial parameters for the normalization utility.
  init: Callable[[], NormalizationParams]
  # Returns the normalized input nested array.
  normalize: Callable[[types.NestedArray, NormalizationParams],
                      types.NestedArray]
  # Returns updates normalization parameters.
  update: Callable[[NormalizationParams, types.NestedArray, Optional[str]],
                   NormalizationParams]


# largely following acme/agents/jax/normalization.py
# TODO(kamyar): move to acme/agents/jax/normalization.py
class NormalizationActorWrapper(core.Actor):
  """An actor wrapper that normalizes observations before applying policy."""

  def __init__(
      self,
      wrapped_actor: core.Actor,
      normalization_fns: NormalizationFns,
      variable_client: variable_utils.VariableClient,
      backend: Optional[str] = None,
  ):
    self._wrapped_actor = wrapped_actor
    self._variable_client = variable_client
    self._apply_normalization = jax.jit(
        normalization_fns.normalize, backend=backend)

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    self._variable_client.update()
    norm_params = self._variable_client.params
    observation = self._apply_normalization(observation, norm_params)
    return self._wrapped_actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    return self._wrapped_actor.observe_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    return self._wrapped_actor.observe(action, next_timestep)

  def update(self, wait: bool = False):
    return self._wrapped_actor.update(wait)


class EMAMeanStdNormalizerParams(NamedTuple):
  """Using technique form Adam optimizer paper for computing running stats."""
  ema_counter: jnp.int32
  biased_first_moment: types.NestedArray
  biased_second_moment: types.NestedArray


def build_ema_mean_std_normalizer(
    nested_spec: types.NestedSpec,
    tau: float = 0.995,
    epsilon: float = 1e-6,) -> NormalizationFns:
  """Builds pure functions used for normalizing based on EMA mean and std.

  The built normalizer functions can be used to normalize nested arrays that
  have a structure corresponding to nested_spec. Currently only supports
  nested_spec where all leafs have float dtype.

  Arguments:
    nested_spec: A nested spec where all leaves have float dtype
    tau: tau parameter for exponential moving average
    epsilon: epsilon for avoiding division by zero std

  Returns:
    NormalizationFns to be used for normalization
  """
  nested_dims = jax.tree_util.tree_map(lambda x: len(x.shape), nested_spec)

  def init() -> EMAMeanStdNormalizerParams:
    first_moment = utils.zeros_like(nested_spec)
    second_moment = utils.zeros_like(nested_spec)

    return EMAMeanStdNormalizerParams(
        ema_counter=jnp.int32(0),
        biased_first_moment=first_moment,
        biased_second_moment=second_moment,
    )

  def _normalize_leaf(x: jnp.array, ema_counter: jnp.int32,
                      biased_first_moment: jnp.array,
                      biased_second_moment: jnp.array) -> jnp.ndarray:
    zero_debias = 1. / (1. - jnp.power(tau, ema_counter))
    mean = biased_first_moment * zero_debias
    second_moment = biased_second_moment * zero_debias
    std = jnp.sqrt(jax.nn.relu(second_moment - mean**2))

    mean = jnp.broadcast_to(mean, x.shape)
    std = jnp.broadcast_to(std, x.shape)
    return (x - mean) / jnp.fmax(std, epsilon)

  def _normalize(nested_array: types.NestedArray,
                 params: EMAMeanStdNormalizerParams) -> types.NestedArray:
    ema_counter = params.ema_counter
    normalized_nested_array = jax.tree_util.tree_map(
        lambda x, f, s: _normalize_leaf(x, ema_counter, f, s),
        nested_array,
        params.biased_first_moment,
        params.biased_second_moment)
    return normalized_nested_array

  def normalize(nested_array: types.NestedArray,
                params: EMAMeanStdNormalizerParams) -> types.NestedArray:
    ema_counter = params.ema_counter
    norm_obs = jax.lax.cond(
        ema_counter > 0,
        _normalize,
        lambda o, p: o,
        nested_array, params)
    return norm_obs

  def _compute_first_moment(x: jnp.ndarray, ndim: int):
    reduce_axes = tuple(range(len(x.shape) - ndim))
    first_moment = jnp.mean(x, axis=reduce_axes)
    return first_moment

  def _compute_second_moment(x: jnp.ndarray, ndim: int):
    reduce_axes = tuple(range(len(x.shape) - ndim))
    second_moment = jnp.mean(x**2, axis=reduce_axes)
    return second_moment

  def update(
      params: EMAMeanStdNormalizerParams,
      nested_array: types.NestedArray,
      pmap_axis_name: Optional[str] = None) -> EMAMeanStdNormalizerParams:
    # compute the stats
    first_moment = jax.tree_util.tree_map(
        _compute_first_moment, nested_array, nested_dims)
    second_moment = jax.tree_util.tree_map(
        _compute_second_moment, nested_array, nested_dims)

    # propagate across devices
    if pmap_axis_name is not None:
      first_moment, second_moment = jax.lax.pmean(
          (first_moment, second_moment), axis_name=pmap_axis_name)

    # update running statistics
    new_first_moment = jax.tree_util.tree_map(
        lambda x, y: tau * x +  # pylint: disable=g-long-lambda
        (1. - tau) * y,
        params.biased_first_moment,
        first_moment)
    new_second_moment = jax.tree_util.tree_map(
        lambda x, y: tau * x +  # pylint: disable=g-long-lambda
        (1. - tau) * y,
        params.biased_second_moment,
        second_moment)

    # update ema_counter and return updated params
    new_params = EMAMeanStdNormalizerParams(
        ema_counter=params.ema_counter + 1,
        biased_first_moment=new_first_moment,
        biased_second_moment=new_second_moment,
    )

    return new_params

  return NormalizationFns(
      init=init,
      normalize=normalize,
      update=update,
  )


def build_mean_std_normalizer(
    nested_spec: types.NestedSpec,
    max_abs_value: Optional[float] = None) -> NormalizationFns:
  """Builds pure functions used for normalizing based on mean and std.

  Arguments:
    nested_spec: A nested spec where all leaves have float dtype
    max_abs_value: Normalized nested arrays will be clipped so that all values
      will be between -max_abs_value and +max_abs_value. Setting to None
      (default) does not perform this clipping.

  Returns:
    NormalizationFns to be used for normalization
  """

  def init() -> RunningStatisticsState:
    return running_statistics.init_state(nested_spec)

  def normalize(
      nested_array: types.NestedArray,
      params: RunningStatisticsState) -> types.NestedArray:
    return running_statistics.normalize(
        nested_array, params, max_abs_value=max_abs_value)

  def update(
      params: RunningStatisticsState,
      nested_array: types.NestedArray,
      pmap_axis_name: Optional[str]) -> RunningStatisticsState:
    return running_statistics.update(
        params, nested_array, pmap_axis_name=pmap_axis_name)

  return NormalizationFns(
      init=init,
      normalize=normalize,
      update=update)
