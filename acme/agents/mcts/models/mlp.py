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

import copy
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
      embedding_size: int,
  ):
    super(MLPTransitionModel, self).__init__(name='mlp_transition_model')

    # Get num actions/observation shape.
    self._num_actions = environment_spec.actions.num_values
    self._input_shape = environment_spec.observations.shape
    self._flat_shape = int(np.prod(self._input_shape))

    # Embedding networks.
    self._state_embedding = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP(hidden_sizes + (embedding_size,), activate_final=True),
    ])
    self._action_embedding = snt.Sequential([
        lambda a: tf.one_hot(a, self._num_actions, dtype=tf.float32),
        snt.Linear(embedding_size),
        tf.nn.relu,
    ])

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

    embedded_state = self._state_embedding(state)
    embedded_action = self._action_embedding(action)

    embedding = embedded_state * embedded_action

    # Predict the next state, reward, and termination.
    next_state = self._state_network(embedding)
    reward = self._reward_network(embedding)
    discount_logits = self._discount_network(embedding)

    return next_state, reward, discount_logits


class MLPModel(base.Model):
  """A simple environment model."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      replay_capacity: int,
      batch_size: int,
      hidden_sizes: Tuple[int, ...],
      embedding_size: int,
      learning_rate: float = 1e-3,
      terminal_tol: float = 1e-3,
  ):
    # Hyperparameters.
    self._batch_size = batch_size
    self._terminal_tol = terminal_tol

    # Modelling
    self._replay = replay.Replay(replay_capacity)
    self._transition_model = MLPTransitionModel(environment_spec, hidden_sizes,
                                                embedding_size)
    self._optimizer = snt.optimizers.Adam(learning_rate)
    self._forward = tf.function(self._transition_model)

    # Model state.
    self._state = None
    self._needs_reset = True
    self._checkpoint = None
    self._oard = None  # [o, a, r, d]

    self._environment_spec = environment_spec

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
      variables = self._transition_model.trainable_variables

      state_loss = tf.square(next_state - o_tp1)
      reward_loss = tf.square(reward - r_t)
      discount_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_t, discount)

      loss = (
          tf.reduce_mean(state_loss) + tf.reduce_mean(reward_loss) +
          tf.reduce_mean(discount_loss))
      gradients = tape.gradient(loss, variables)

    self._optimizer.apply(gradients, variables)

    return loss

  def step(self, action: types.Action):
    # Reset if required.
    if self._needs_reset:
      raise ValueError('Model must be reset with an initial timestep.')

    # Step the model.
    state, action = tf2_utils.add_batch_dim([self._state, action])
    new_state, reward, discount_logits = self._forward(state, action)
    new_state = tf.squeeze(new_state, axis=0)
    reward = tf.squeeze(reward, axis=0)
    discount_logits = tf.squeeze(discount_logits, axis=0)
    discount = special.softmax(discount_logits)

    # Save the resulting state for the next step.
    self._state = new_state.numpy()

    # We threshold discount on a given tolerance.
    if discount < self._terminal_tol:
      timestep = dm_env.termination(reward=reward, observation=state.copy())
      self._needs_reset = True
    else:
      timestep = dm_env.transition(reward=reward, observation=state.copy())

    return timestep

  def reset(self, initial_state: types.Observation = None):
    if initial_state is None:
      raise ValueError('Model must be reset with an initial state.')
    # We reset to an initial state that we are explicitly given.
    # This allows us to handle environments with stochastic resets (e.g. Catch).
    self._state = copy.deepcopy(initial_state)
    self._needs_reset = False
    return dm_env.restart(self._state)

  def update_last(self, observation: types.Observation):
    if not self._oard:
      raise ValueError('Empty buffer')
    self._replay.add(self._oard + [observation])
    self._oard = None
    self._needs_reset = True

  def update(
      self,
      observation: types.Observation,
      action: types.Action,
      reward: types.Reward,
      discount: types.Discount,
  ):
    if self._oard:
      self._replay.add(self._oard + [observation])
    self._oard = [observation, action, reward, discount]

    # Step the model to keep in sync with 'reality'.
    ts = self.step(action)

    if ts.last():
      # Model believes that a termination has happened.
      # This will result in a crash during planning if the true environment
      # didn't terminate here as well. So, we indicate that we need a reset.
      self._needs_reset = True

    # Sample from replay and do SGD.
    if self._replay.size >= self._batch_size:
      batch = self._replay.sample(self._batch_size)
      self._step(*batch)

  def save_checkpoint(self):
    if self._needs_reset:
      raise ValueError('Cannot save checkpoint: model must be reset first.')
    self._checkpoint = self._state.copy()  # type: types.Observation

  def load_checkpoint(self):
    self._needs_reset = False
    self._state = self._checkpoint.copy()  # type: types.Observation

  def action_spec(self):
    return self._environment_spec.actions

  def observation_spec(self):
    return self._environment_spec.observations

  @property
  def needs_reset(self) -> bool:
    return self._needs_reset
