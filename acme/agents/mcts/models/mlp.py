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

"""A simple (deterministic) environment transition model from pixels."""

from typing import Tuple

from acme import specs
from acme.agents.mcts import types
from acme.agents.mcts.models import base
from acme.utils import tf2_utils

from bsuite.baselines.utils import replay
import dm_env
import numpy as np
from scipy import special
import sonnet as snt
import tensorflow as tf


class MLPTransitionModel(snt.Module):
  """This uses MLPs to model (s, a) -> (r, d, s')."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      hidden_sizes: Tuple[int, ...],
  ):
    super(MLPTransitionModel, self).__init__(name='mlp_transition_model')

    # Get num actions/observation shape.
    self._num_actions = environment_spec.actions.num_values
    self._input_shape = environment_spec.observations.shape
    self._flat_shape = int(np.prod(self._input_shape))

    # Prediction networks.
    self._state_network = snt.Sequential([
        snt.nets.MLP(hidden_sizes + (self._flat_shape,)),
        snt.Reshape(self._input_shape)
    ])
    self._reward_network = snt.Sequential([
        snt.nets.MLP(hidden_sizes + (1,)),
        lambda r: tf.squeeze(r, axis=-1),
    ])
    self._discount_network = snt.Sequential([
        snt.nets.MLP(hidden_sizes + (1,)),
        lambda d: tf.squeeze(d, axis=-1),
    ])

  def __call__(self, state: tf.Tensor,
               action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    embedded_state = snt.Flatten()(state)
    embedded_action = tf.one_hot(action, depth=self._num_actions)

    embedding = tf.concat([embedded_state, embedded_action], axis=-1)

    # Predict the next state, reward, and termination.
    next_state = self._state_network(embedding)
    reward = self._reward_network(embedding)
    discount_logits = self._discount_network(embedding)

    return next_state, reward, discount_logits


class MLPModel(base.Model):
  """A simple environment model."""

  _checkpoint: types.Observation
  _state: types.Observation

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      replay_capacity: int,
      batch_size: int,
      hidden_sizes: Tuple[int, ...],
      learning_rate: float = 1e-3,
      terminal_tol: float = 1e-3,
  ):
    self._obs_spec = environment_spec.observations
    self._action_spec = environment_spec.actions
    # Hyperparameters.
    self._batch_size = batch_size
    self._terminal_tol = terminal_tol

    # Modelling
    self._replay = replay.Replay(replay_capacity)
    self._transition_model = MLPTransitionModel(environment_spec, hidden_sizes)
    self._optimizer = snt.optimizers.Adam(learning_rate)
    self._forward = tf.function(self._transition_model)
    tf2_utils.create_variables(
        self._transition_model, [self._obs_spec, self._action_spec])
    self._variables = self._transition_model.trainable_variables

    # Model state.
    self._needs_reset = True

  @tf.function
  def _step(
      self,
      o_t: tf.Tensor,
      a_t: tf.Tensor,
      r_t: tf.Tensor,
      d_t: tf.Tensor,
      o_tp1: tf.Tensor,
  ) -> tf.Tensor:

    with tf.GradientTape() as tape:
      next_state, reward, discount = self._transition_model(o_t, a_t)

      state_loss = tf.square(next_state - o_tp1)
      reward_loss = tf.square(reward - r_t)
      discount_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_t, discount)

      loss = sum([
          tf.reduce_mean(state_loss),
          tf.reduce_mean(reward_loss),
          tf.reduce_mean(discount_loss),
      ])

    gradients = tape.gradient(loss, self._variables)
    self._optimizer.apply(gradients, self._variables)

    return loss

  def step(self, action: types.Action):
    # Reset if required.
    if self._needs_reset:
      raise ValueError('Model must be reset with an initial timestep.')

    # Step the model.
    state, action = tf2_utils.add_batch_dim([self._state, action])
    new_state, reward, discount_logits = [
        x.numpy().squeeze(axis=0) for x in self._forward(state, action)
    ]
    discount = special.softmax(discount_logits)

    # Save the resulting state for the next step.
    self._state = new_state

    # We threshold discount on a given tolerance.
    if discount < self._terminal_tol:
      self._needs_reset = True
      return dm_env.termination(reward=reward, observation=self._state.copy())
    return dm_env.transition(reward=reward, observation=self._state.copy())

  def reset(self, initial_state: types.Observation = None):
    if initial_state is None:
      raise ValueError('Model must be reset with an initial state.')
    # We reset to an initial state that we are explicitly given.
    # This allows us to handle environments with stochastic resets (e.g. Catch).
    self._state = initial_state.copy()
    self._needs_reset = False
    return dm_env.restart(self._state)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: types.Action,
      next_timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    # Add the true transition to replay.
    transition = [
        timestep.observation,
        action,
        next_timestep.reward,
        next_timestep.discount,
        next_timestep.observation,
    ]
    self._replay.add(transition)

    # Step the model to generate a synthetic transition.
    ts = self.step(action)

    # Copy the *true* state on update.
    self._state = next_timestep.observation.copy()

    if ts.last() or next_timestep.last():
      # Model believes that a termination has happened.
      # This will result in a crash during planning if the true environment
      # didn't terminate here as well. So, we indicate that we need a reset.
      self._needs_reset = True

    # Sample from replay and do SGD.
    if self._replay.size >= self._batch_size:
      batch = self._replay.sample(self._batch_size)
      self._step(*batch)

    return ts

  def save_checkpoint(self):
    if self._needs_reset:
      raise ValueError('Cannot save checkpoint: model must be reset first.')
    self._checkpoint = self._state.copy()

  def load_checkpoint(self):
    self._needs_reset = False
    self._state = self._checkpoint.copy()

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._obs_spec

  @property
  def needs_reset(self) -> bool:
    return self._needs_reset
