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

"""AIL discriminator losses."""

import functools
from typing import Callable, Dict, Optional, Tuple

from acme import types
from acme.jax import networks as networks_lib
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
import tree

tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

# The loss is a function taking the discriminator, its state, the demo
# transition and the replay buffer transitions.
# It returns the loss as a float and a debug dictionary with the new state.
State = networks_lib.Params
DiscriminatorOutput = Tuple[networks_lib.Logits, State]
DiscriminatorFn = Callable[[State, types.Transition], DiscriminatorOutput]
Metrics = Dict[str, float]
LossOutput = Tuple[float, Tuple[Metrics, State]]
Loss = Callable[[
    DiscriminatorFn, State, types.Transition, types.Transition, networks_lib
    .PRNGKey
], LossOutput]


def _binary_cross_entropy_loss(logit: jnp.ndarray,
                               label: jnp.ndarray) -> jnp.ndarray:
  return label * jax.nn.softplus(-logit) + (1 - label) * jax.nn.softplus(logit)


@jax.vmap
def _weighted_average(x: jnp.ndarray, y: jnp.ndarray,
                      lambdas: jnp.ndarray) -> jnp.ndarray:
  return lambdas * x + (1. - lambdas) * y


def _label_data(
    rb_transitions: types.Transition,
    demonstration_transitions: types.Transition, mixup_alpha: Optional[float],
    key: networks_lib.PRNGKey) -> Tuple[types.Transition, jnp.ndarray]:
  """Create a tuple data, labels by concatenating the rb and dem transitions."""
  data = tree.map_structure(lambda x, y: jnp.concatenate([x, y]),
                            rb_transitions, demonstration_transitions)
  labels = jnp.concatenate([
      jnp.zeros(rb_transitions.reward.shape),
      jnp.ones(demonstration_transitions.reward.shape)
  ])

  if mixup_alpha is not None:
    lambda_key, mixup_key = jax.random.split(key)

    lambdas = tfd.Beta(mixup_alpha, mixup_alpha).sample(
        len(labels), seed=lambda_key)

    shuffled_data = tree.map_structure(
        lambda x: jax.random.permutation(key=mixup_key, x=x), data)
    shuffled_labels = jax.random.permutation(key=mixup_key, x=labels)

    data = tree.map_structure(lambda x, y: _weighted_average(x, y, lambdas),
                              data, shuffled_data)
    labels = _weighted_average(labels, shuffled_labels, lambdas)

  return data, labels


def _logit_bernoulli_entropy(logits: networks_lib.Logits) -> jnp.ndarray:
  return (1. - jax.nn.sigmoid(logits)) * logits - jax.nn.log_sigmoid(logits)


def gail_loss(entropy_coefficient: float = 0.,
              mixup_alpha: Optional[float] = None) -> Loss:
  """Computes the standard GAIL loss."""

  def loss_fn(
      discriminator_fn: DiscriminatorFn,
      discriminator_state: State,
      demo_transitions: types.Transition, rb_transitions: types.Transition,
      rng_key: networks_lib.PRNGKey) -> LossOutput:

    data, labels = _label_data(
        rb_transitions=rb_transitions,
        demonstration_transitions=demo_transitions,
        mixup_alpha=mixup_alpha,
        key=rng_key)
    logits, discriminator_state = discriminator_fn(discriminator_state, data)

    classification_loss = jnp.mean(_binary_cross_entropy_loss(logits, labels))

    entropy = jnp.mean(_logit_bernoulli_entropy(logits))
    entropy_loss = -entropy_coefficient * entropy

    total_loss = classification_loss + entropy_loss

    metrics = {
        'total_loss': total_loss,
        'entropy_loss': entropy_loss,
        'classification_loss': classification_loss
    }
    return total_loss, (metrics, discriminator_state)

  return loss_fn


def pugail_loss(positive_class_prior: float,
                entropy_coefficient: float,
                pugail_beta: Optional[float] = None) -> Loss:
  """Computes the PUGAIL loss (https://arxiv.org/pdf/1911.00459.pdf)."""

  def loss_fn(
      discriminator_fn: DiscriminatorFn,
      discriminator_state: State,
      demo_transitions: types.Transition, rb_transitions: types.Transition,
      rng_key: networks_lib.PRNGKey) -> LossOutput:
    del rng_key

    demo_logits, discriminator_state = discriminator_fn(discriminator_state,
                                                        demo_transitions)
    rb_logits, discriminator_state = discriminator_fn(discriminator_state,
                                                      rb_transitions)

    # Quick Maths:
    # output = logit(D) = ln(D) - ln(1-D)
    # -softplus(-output) = ln(D)
    # softplus(output) = -ln(1-D)

    # prior * -ln(D(expert))
    positive_loss = positive_class_prior * -jax.nn.log_sigmoid(demo_logits)
    # -ln(1 - D(policy)) - prior * -ln(1 - D(expert))
    negative_loss = jax.nn.softplus(
        rb_logits) - positive_class_prior * jax.nn.softplus(demo_logits)
    if pugail_beta is not None:
      negative_loss = jnp.clip(negative_loss, a_min=-1. * pugail_beta)

    classification_loss = jnp.mean(positive_loss + negative_loss)

    entropy = jnp.mean(
        _logit_bernoulli_entropy(jnp.concatenate([demo_logits, rb_logits])))
    entropy_loss = -entropy_coefficient * entropy

    total_loss = classification_loss + entropy_loss

    metrics = {
        'total_loss': total_loss,
        'positive_loss': jnp.mean(positive_loss),
        'negative_loss': jnp.mean(negative_loss),
        'demo_logits': jnp.mean(demo_logits),
        'rb_logits': jnp.mean(rb_logits),
        'entropy_loss': entropy_loss,
        'classification_loss': classification_loss
    }
    return total_loss, (metrics, discriminator_state)

  return loss_fn


def _make_gradient_penalty_data(rb_transitions: types.Transition,
                                demonstration_transitions: types.Transition,
                                key: networks_lib.PRNGKey) -> types.Transition:
  lambdas = tfd.Uniform().sample(len(rb_transitions.reward), seed=key)
  return tree.map_structure(lambda x, y: _weighted_average(x, y, lambdas),
                            rb_transitions, demonstration_transitions)


@functools.partial(jax.vmap, in_axes=(0, None, None))
def _compute_gradient_penalty(gradient_penalty_data: types.Transition,
                              discriminator_fn: Callable[[types.Transition],
                                                         float],
                              gradient_penalty_target: float) -> float:
  """Computes a penalty based on the gradient norm on the data."""
  # The input should not be batched.
  assert not gradient_penalty_data.reward.shape
  discriminator_gradient_fn = jax.grad(discriminator_fn)
  gradients = discriminator_gradient_fn(gradient_penalty_data)
  gradients = tree.map_structure(lambda x: x.flatten(), gradients)
  gradients = jnp.concatenate([gradients.observation, gradients.action,
                               gradients.next_observation])
  gradient_norms = jnp.linalg.norm(gradients + 1e-8)
  k = gradient_penalty_target * jnp.ones_like(gradient_norms)
  return jnp.mean(jnp.square(gradient_norms - k))


def add_gradient_penalty(base_loss: Loss,
                         gradient_penalty_coefficient: float,
                         gradient_penalty_target: float) -> Loss:
  """Adds a gradient penalty to the base_loss."""

  if not gradient_penalty_coefficient:
    return base_loss

  def loss_fn(discriminator_fn: DiscriminatorFn,
              discriminator_state: State,
              demo_transitions: types.Transition,
              rb_transitions: types.Transition,
              rng_key: networks_lib.PRNGKey) -> LossOutput:
    super_key, gradient_penalty_key = jax.random.split(rng_key)

    partial_loss, (losses, discriminator_state) = base_loss(
        discriminator_fn, discriminator_state, demo_transitions, rb_transitions,
        super_key)

    gradient_penalty_data = _make_gradient_penalty_data(
        rb_transitions=rb_transitions,
        demonstration_transitions=demo_transitions,
        key=gradient_penalty_key)
    def apply_discriminator_fn(transitions: types.Transition) -> float:
      logits, _ = discriminator_fn(discriminator_state, transitions)
      return logits
    gradient_penalty = gradient_penalty_coefficient * jnp.mean(
        _compute_gradient_penalty(gradient_penalty_data, apply_discriminator_fn,
                                  gradient_penalty_target))

    losses['gradient_penalty'] = gradient_penalty
    total_loss = partial_loss + gradient_penalty
    losses['total_loss'] = total_loss

    return total_loss, (losses, discriminator_state)

  return loss_fn
