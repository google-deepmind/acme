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

"""ARS learner implementation."""

import collections
import threading
import time
from typing import Any, Deque, Dict, Iterator, List, NamedTuple, Optional

import acme
from acme import specs
from acme.adders import reverb as acme_reverb
from acme.agents.jax.ars import config as ars_config
from acme.agents.jax.ars import networks as ars_networks
from acme.jax import networks as networks_lib
from acme.jax import running_statistics
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import numpy as np
import reverb


class PerturbationKey(NamedTuple):
  training_iteration: int
  perturbation_id: int
  is_opposite: bool


class EvaluationResult(NamedTuple):
  total_reward: float
  observation: networks_lib.Observation


class EvaluationRequest(NamedTuple):
  key: PerturbationKey
  policy_params: networks_lib.Params
  normalization_params: networks_lib.Params


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  key: networks_lib.PRNGKey
  normalizer_params: networks_lib.Params
  policy_params: networks_lib.Params
  training_iteration: int


class EvaluationState(NamedTuple):
  """Contains training state for the learner."""
  key: networks_lib.PRNGKey
  evaluation_queue: Deque[EvaluationRequest]
  received_results: Dict[PerturbationKey, EvaluationResult]
  noises: List[networks_lib.Params]


class ARSLearner(acme.Learner):
  """ARS learner."""

  _state: TrainingState

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      networks: networks_lib.FeedForwardNetwork,
      rng: networks_lib.PRNGKey,
      config: ars_config.ARSConfig,
      iterator: Iterator[reverb.ReplaySample],
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None):

    self._config = config
    self._lock = threading.Lock()

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key=self._counter.get_steps_key())

    # Iterator on demonstration transitions.
    self._iterator = iterator

    if self._config.normalize_observations:
      normalizer_params = running_statistics.init_state(spec.observations)
      self._normalizer_update_fn = running_statistics.update
    else:
      normalizer_params = ()
      self._normalizer_update_fn = lambda a, b: a

    rng1, rng2, tmp = jax.random.split(rng, 3)
    # Create initial state.
    self._training_state = TrainingState(
        key=rng1,
        policy_params=networks.init(tmp),
        normalizer_params=normalizer_params,
        training_iteration=0)
    self._evaluation_state = EvaluationState(
        key=rng2,
        evaluation_queue=collections.deque(),
        received_results={},
        noises=[])

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def _generate_perturbations(self):
    with self._lock:
      rng, noise_key = jax.random.split(self._evaluation_state.key)
      self._evaluation_state = EvaluationState(
          key=rng,
          evaluation_queue=collections.deque(),
          received_results={},
          noises=[])

      all_noise = jax.random.normal(
          noise_key,
          shape=(self._config.num_directions,) +
          self._training_state.policy_params.shape,
          dtype=self._training_state.policy_params.dtype)
      for i in range(self._config.num_directions):
        noise = all_noise[i]
        self._evaluation_state.noises.append(noise)
        for direction in (-1, 1):
          self._evaluation_state.evaluation_queue.append(
              EvaluationRequest(
                  PerturbationKey(self._training_state.training_iteration, i,
                                  direction == -1),
                  self._training_state.policy_params +
                  direction * noise * self._config.exploration_noise_std,
                  self._training_state.normalizer_params))

  def _read_results(self):
    while len(self._evaluation_state.received_results
             ) != self._config.num_directions * 2:
      data = next(self._iterator).data
      data = acme_reverb.Step(*data)

      # validation
      params_key = data.extras['params_key']
      training_step, perturbation_id, is_opposite = params_key
      # If the incoming data does not correspond to the current iteration,
      # we simply ignore it.
      if not np.all(
          training_step[:-1] == self._training_state.training_iteration):
        continue

      # The whole episode should be run with the same policy, so let's check
      # for that.
      assert np.all(perturbation_id[:-1] == perturbation_id[0])
      assert np.all(is_opposite[:-1] == is_opposite[0])

      perturbation_id = perturbation_id[0].item()
      is_opposite = is_opposite[0].item()

      total_reward = np.sum(data.reward - self._config.reward_shift)
      k = PerturbationKey(self._training_state.training_iteration,
                          perturbation_id, is_opposite)
      if k in self._evaluation_state.received_results:
        continue
      self._evaluation_state.received_results[k] = EvaluationResult(
          total_reward, data.observation)

  def _update_model(self) -> int:
    # Update normalization params.
    real_actor_steps = 0
    normalizer_params = self._training_state.normalizer_params
    for _, value in self._evaluation_state.received_results.items():
      real_actor_steps += value.observation.shape[0] - 1
      normalizer_params = self._normalizer_update_fn(normalizer_params,
                                                     value.observation)

    # Keep only top directions.
    top_directions = []
    for i in range(self._config.num_directions):
      reward_forward = self._evaluation_state.received_results[PerturbationKey(
          self._training_state.training_iteration, i, False)].total_reward
      reward_reverse = self._evaluation_state.received_results[PerturbationKey(
          self._training_state.training_iteration, i, True)].total_reward
      top_directions.append((max(reward_forward, reward_reverse), i))
    top_directions.sort()
    top_directions = top_directions[-self._config.top_directions:]

    # Compute reward_std.
    reward = []
    for _, i in top_directions:
      reward.append(self._evaluation_state.received_results[PerturbationKey(
          self._training_state.training_iteration, i, False)].total_reward)
      reward.append(self._evaluation_state.received_results[PerturbationKey(
          self._training_state.training_iteration, i, True)].total_reward)
    reward_std = np.std(reward)

    # Compute new policy params.
    policy_params = self._training_state.policy_params
    curr_sum = np.zeros_like(policy_params)
    for _, i in top_directions:
      reward_forward = self._evaluation_state.received_results[PerturbationKey(
          self._training_state.training_iteration, i, False)].total_reward
      reward_reverse = self._evaluation_state.received_results[PerturbationKey(
          self._training_state.training_iteration, i, True)].total_reward
      curr_sum += self._evaluation_state.noises[i] * (
          reward_forward - reward_reverse)

    policy_params = policy_params + self._config.step_size / (
        self._config.top_directions * reward_std) * curr_sum

    self._training_state = TrainingState(
        key=self._training_state.key,
        normalizer_params=normalizer_params,
        policy_params=policy_params,
        training_iteration=self._training_state.training_iteration)
    return real_actor_steps

  def step(self):
    self._training_state = self._training_state._replace(
        training_iteration=self._training_state.training_iteration + 1)
    self._generate_perturbations()
    self._read_results()
    real_actor_steps = self._update_model()

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(
        steps=1,
        real_actor_steps=real_actor_steps,
        learner_episodes=2 * self._config.num_directions,
        walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write(counts)

  def get_variables(self, names: List[str]) -> List[Any]:
    assert (names == [ars_networks.BEHAVIOR_PARAMS_NAME] or
            names == [ars_networks.EVAL_PARAMS_NAME])
    if names == [ars_networks.EVAL_PARAMS_NAME]:
      return [PerturbationKey(-1, -1, False),
              self._training_state.policy_params,
              self._training_state.normalizer_params]
    should_sleep = False
    while True:
      if should_sleep:
        time.sleep(0.1)
      should_sleep = False
      with self._lock:
        if not self._evaluation_state.evaluation_queue:
          should_sleep = True
          continue
        data = self._evaluation_state.evaluation_queue.pop()
        # If this perturbation was already evaluated, we simply skip it.
        if data.key in self._evaluation_state.received_results:
          continue
        # In case if an actor fails we still need to reevaluate the same
        # perturbation, so we just add it to the end of the queue.
        self._evaluation_state.evaluation_queue.append(data)
        return [data]

  def save(self) -> TrainingState:
    return self._training_state

  def restore(self, state: TrainingState):
    self._training_state = state
