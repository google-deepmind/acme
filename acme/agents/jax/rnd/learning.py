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

"""RND learner implementation."""

import functools
import time
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.agents.jax.rnd import networks as rnd_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import reverb_utils
import jax
import jax.numpy as jnp
import optax
import reverb


class RNDTrainingState(NamedTuple):
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: networks_lib.Params
  target_params: networks_lib.Params
  steps: int


class GlobalTrainingState(NamedTuple):
  """Contains training state of the RND learner."""
  rewarder_state: RNDTrainingState
  learner_state: Any


RNDLoss = Callable[[networks_lib.Params, networks_lib.Params, types.Transition],
                   float]


def rnd_update_step(
    state: RNDTrainingState, transitions: types.Transition,
    loss_fn: RNDLoss, optimizer: optax.GradientTransformation
) -> Tuple[RNDTrainingState, Dict[str, jnp.ndarray]]:
  """Run an update steps on the given transitions.

  Args:
    state: The learner state.
    transitions: Transitions to update on.
    loss_fn: The loss function.
    optimizer: The optimizer of the predictor network.

  Returns:
    A new state and metrics.
  """
  loss, grads = jax.value_and_grad(loss_fn)(
      state.params,
      state.target_params,
      transitions=transitions)

  update, optimizer_state = optimizer.update(grads, state.optimizer_state)
  params = optax.apply_updates(state.params, update)

  new_state = RNDTrainingState(
      optimizer_state=optimizer_state,
      params=params,
      target_params=state.target_params,
      steps=state.steps + 1,
  )
  return new_state, {'rnd_loss': loss}


def rnd_loss(
    predictor_params: networks_lib.Params,
    target_params: networks_lib.Params,
    transitions: types.Transition,
    networks: rnd_networks.RNDNetworks,
) -> float:
  """The Random Network Distillation loss.

  See https://arxiv.org/pdf/1810.12894.pdf A.2

  Args:
    predictor_params: Parameters of the predictor
    target_params: Parameters of the target
    transitions: Transitions to compute the loss on.
    networks: RND networks

  Returns:
    The MSE loss as a float.
  """
  target_output = networks.target.apply(target_params,
                                        transitions.observation,
                                        transitions.action)
  predictor_output = networks.predictor.apply(predictor_params,
                                              transitions.observation,
                                              transitions.action)
  return jnp.mean(jnp.square(target_output - predictor_output))


class RNDLearner(acme.Learner):
  """RND learner."""

  def __init__(
      self,
      direct_rl_learner_factory: Callable[[Any, Iterator[reverb.ReplaySample]],
                                          acme.Learner],
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      rnd_network: rnd_networks.RNDNetworks,
      rng_key: jnp.ndarray,
      grad_updates_per_batch: int,
      is_sequence_based: bool,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None):
    self._is_sequence_based = is_sequence_based

    target_key, predictor_key = jax.random.split(rng_key)
    target_params = rnd_network.target.init(target_key)
    predictor_params = rnd_network.predictor.init(predictor_key)
    optimizer_state = optimizer.init(predictor_params)

    self._state = RNDTrainingState(
        optimizer_state=optimizer_state,
        params=predictor_params,
        target_params=target_params,
        steps=0)

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key=self._counter.get_steps_key())

    loss = functools.partial(rnd_loss, networks=rnd_network)
    self._update = functools.partial(rnd_update_step,
                                     loss_fn=loss,
                                     optimizer=optimizer)
    self._update = utils.process_multiple_batches(self._update,
                                                  grad_updates_per_batch)
    self._update = jax.jit(self._update)

    self._get_reward = jax.jit(
        functools.partial(
            rnd_networks.compute_rnd_reward, networks=rnd_network))

    # Generator expression that works the same as an iterator.
    # https://pymbook.readthedocs.io/en/latest/igd.html#generator-expressions
    updated_iterator = (self._process_sample(sample) for sample in iterator)

    self._direct_rl_learner = direct_rl_learner_factory(
        rnd_network.direct_rl_networks, updated_iterator)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def _process_sample(self, sample: reverb.ReplaySample) -> reverb.ReplaySample:
    """Uses the replay sample to train and update its reward.

    Args:
      sample: Replay sample to train on.

    Returns:
      The sample replay sample with an updated reward.
    """
    transitions = reverb_utils.replay_sample_to_sars_transition(
        sample, is_sequence=self._is_sequence_based)
    self._state, metrics = self._update(self._state, transitions)
    rewards = self._get_reward(self._state.params, self._state.target_params,
                               transitions)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

    return sample._replace(data=sample.data._replace(reward=rewards))

  def step(self):
    self._direct_rl_learner.step()

  def get_variables(self, names: List[str]) -> List[Any]:
    rnd_variables = {
        'target_params': self._state.target_params,
        'predictor_params': self._state.params
    }

    learner_names = [name for name in names if name not in rnd_variables]
    learner_dict = {}
    if learner_names:
      learner_dict = dict(
          zip(learner_names,
              self._direct_rl_learner.get_variables(learner_names)))

    variables = [
        rnd_variables.get(name, learner_dict.get(name, None)) for name in names
    ]
    return variables

  def save(self) -> GlobalTrainingState:
    return GlobalTrainingState(
        rewarder_state=self._state,
        learner_state=self._direct_rl_learner.save())

  def restore(self, state: GlobalTrainingState):
    self._state = state.rewarder_state
    self._direct_rl_learner.restore(state.learner_state)
