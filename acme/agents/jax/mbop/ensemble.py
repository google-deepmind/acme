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

"""Module to provide ensembling support on top of a base network."""
import functools
from typing import (Any, Callable)

from acme.jax import networks
import jax
import jax.numpy as jnp


def _split_batch_dimension(new_batch: int, data: jnp.ndarray) -> jnp.ndarray:
  """Splits the batch dimension and introduces new one with size `new_batch`.

  The result has two batch dimensions, first one of size `new_batch`, second one
  of size `data.shape[0]/new_batch`. It expects that `data.shape[0]` is
  divisible by `new_batch`.

  Args:
    new_batch: Dimension of outer batch dimension.
    data: jnp.ndarray to be reshaped.

  Returns:
    jnp.ndarray with extra batch dimension at start and updated second
    dimension.
  """
  # The first dimension will be used for allocating to a specific ensemble
  # member, and the second dimension is the parallelized batch dimension, and
  # the remaining dimensions are passed as-is to the wrapped network.
  # We use Fortan (F) order so that each input batch i is allocated to
  # ensemble member k = i % new_batch.
  return jnp.reshape(data, (new_batch, -1) + data.shape[1:], order='F')


def _repeat_n(new_batch: int, data: jnp.ndarray) -> jnp.ndarray:
  """Create new batch dimension of size `new_batch` by repeating `data`."""
  return jnp.broadcast_to(data, (new_batch,) + data.shape)


def ensemble_init(base_init: Callable[[networks.PRNGKey], networks.Params],
                  num_networks: int, rnd: jnp.ndarray):
  """Initializes the ensemble parameters.

  Args:
    base_init: An init function that takes only a PRNGKey, if a network's init
      function requires other parameters such as example inputs they need to
      have been previously wrapped i.e. with functool.partial using kwargs.
    num_networks: Number of networks to generate parameters for.
    rnd: PRNGKey to split from when generating parameters.

  Returns:
    `params` for the set of ensemble networks.
  """
  rnds = jax.random.split(rnd, num_networks)
  return jax.vmap(base_init)(rnds)


def apply_round_robin(base_apply: Callable[[networks.Params, Any], Any],
                      params: networks.Params, *args, **kwargs) -> Any:
  """Passes the input in a round-robin manner.

  The round-robin application means that each element of the input batch will
  be passed through a single ensemble member in a deterministic round-robin
  manner, i.e. element_i -> member_k where k = i % num_networks.

  It expects that:
  * `base_apply(member_params, *member_args, **member_kwargs)` is a valid call,
     where:
    * `member_params.shape = params.shape[1:]`
    * `member_args` and `member_kwargs` have the same structure as `args` and
      `kwargs`.
  * `params[k]` contains the params of the k-th member of the ensemble.
  * All jax arrays in `args` and `kwargs` have a batch dimension at axis 0 of
    the same size, which is divisible by `params.shape[0]`.

  Args:
    base_apply: Base network `apply` function that will be used for round-robin.
      NOTE -- This will not work with mutable/stateful apply functions. --
    params: Model parameters.  Number of networks is deduced from this.
    *args: Allows for arbitrary call signatures for `base_apply`.
    **kwargs: Allows for arbitrary call signatures for `base_apply`.

  Returns:
    pytree of the round-robin application.
    Output shape will be [initial_batch_size, <remaining dimensions>].
  """
  # `num_networks` is the size of the batch dimension in `params`.
  num_networks = jax.tree_util.tree_leaves(params)[0].shape[0]

  # Reshape args and kwargs for the round-robin:
  args = jax.tree_map(
      functools.partial(_split_batch_dimension, num_networks), args)
  kwargs = jax.tree_map(
      functools.partial(_split_batch_dimension, num_networks), kwargs)
  # `out.shape` is `(num_networks, initial_batch_size/num_networks, ...)
  out = jax.vmap(base_apply)(params, *args, **kwargs)
  # Reshape to [initial_batch_size, <remaining dimensions>]. Using the 'F' order
  # forces the original values to the last dimension.
  return jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:], order='F'), out)


def apply_all(base_apply: Callable[[networks.Params, Any], Any],
              params: networks.Params, *args, **kwargs) -> Any:
  """Pass the input to all ensemble members.

  Inputs can either have a batch dimension which will get implicitly vmapped
  over, or can be a single vector which will get sent to all ensemble members.
  e.g. [<inputs_dims>] or [batch_size, <input_dims>].

  Args:
    base_apply: Base network `apply` function that will be used for averaging.
      NOTE -- This will not work with mutable/stateful apply functions. --
    params: Model parameters.  Number of networks is deduced from this.
    *args: Allows for arbitrary call signatures for `base_apply`.
    **kwargs: Allows for arbitrary call signatures for `base_apply`.

  Returns:
    pytree of the resulting output of passing input to all ensemble members.
    Output shape will be [num_members, batch_size, <network output dims>].
  """
  # `num_networks` is the size of the batch dimension in `params`.
  num_networks = jax.tree_util.tree_leaves(params)[0].shape[0]

  args = jax.tree_map(functools.partial(_repeat_n, num_networks), args)
  kwargs = jax.tree_map(functools.partial(_repeat_n, num_networks), kwargs)
  # `out` is of shape `(num_networks, batch_size, <remaining dimensions>)`.
  return jax.vmap(base_apply)(params, *args, **kwargs)


def apply_mean(base_apply: Callable[[networks.Params, Any], Any],
               params: networks.Params, *args, **kwargs) -> Any:
  """Calculates the mean over all ensemble members for each batch element.

  Args:
    base_apply: Base network `apply` function that will be used for averaging.
      NOTE -- This will not work with mutable/stateful apply functions. --
    params: Model parameters.  Number of networks is deduced from this.
    *args: Allows for arbitrary call signatures for `base_apply`.
    **kwargs: Allows for arbitrary call signatures for `base_apply`.

  Returns:
    pytree of the average over all ensembles for each element.
    Output shape will be [batch_size, <network output_dims>]
  """
  out = apply_all(base_apply, params, *args, **kwargs)
  return jax.tree_map(functools.partial(jnp.mean, axis=0), out)


def make_ensemble(base_network: networks.FeedForwardNetwork,
                  ensemble_apply: Callable[..., Any],
                  num_networks: int) -> networks.FeedForwardNetwork:
  return networks.FeedForwardNetwork(
      init=functools.partial(ensemble_init, base_network.init, num_networks),
      apply=functools.partial(ensemble_apply, base_network.apply))
