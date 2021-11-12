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

"""AIL learner implementation."""
import functools
import itertools
import time
from typing import Any, Callable, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.agents.jax.ail import losses
from acme.agents.jax.ail import networks as ail_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import reverb_utils
import jax
import optax
import reverb


class DiscriminatorTrainingState(NamedTuple):
  """Contains training state for the discriminator."""
  # State of the optimizer used to optimize the discriminator parameters.
  optimizer_state: optax.OptState

  # Parameters of the discriminator.
  discriminator_params: networks_lib.Params

  # State of the discriminator
  discriminator_state: losses.State

  # For AIRL variants, we need the policy params to compute the loss.
  policy_params: Optional[networks_lib.Params]

  # Key for random number generation.
  key: networks_lib.PRNGKey

  # Training step of the discriminator.
  steps: int


class TrainingState(NamedTuple):
  """Contains training state of the AIL learner."""
  rewarder_state: DiscriminatorTrainingState
  learner_state: Any


def ail_update_step(
    state: DiscriminatorTrainingState, data: Tuple[types.Transition,
                                                   types.Transition],
    optimizer: optax.GradientTransformation,
    ail_network: ail_networks.AILNetworks,
    loss_fn: losses.Loss) -> Tuple[DiscriminatorTrainingState, losses.Metrics]:
  """Run an update steps on the given transitions.

  Args:
    state: The learner state.
    data: Demo and rb transitions.
    optimizer: Discriminator optimizer.
    ail_network: AIL networks.
    loss_fn: Discriminator loss to minimize.

  Returns:
    A new state and metrics.
  """
  demo_transitions, rb_transitions = data
  key, discriminator_key, loss_key = jax.random.split(state.key, 3)

  def compute_loss(
      discriminator_params: networks_lib.Params) -> losses.LossOutput:
    discriminator_fn = functools.partial(
        ail_network.discriminator_network.apply,
        discriminator_params,
        state.policy_params,
        is_training=True,
        rng=discriminator_key)
    return loss_fn(discriminator_fn, state.discriminator_state,
                   demo_transitions, rb_transitions, loss_key)

  loss_grad = jax.grad(compute_loss, has_aux=True)

  grads, (loss, new_discriminator_state) = loss_grad(state.discriminator_params)

  update, optimizer_state = optimizer.update(
      grads,
      state.optimizer_state,
      params=state.discriminator_params)
  discriminator_params = optax.apply_updates(state.discriminator_params, update)

  new_state = DiscriminatorTrainingState(
      optimizer_state=optimizer_state,
      discriminator_params=discriminator_params,
      discriminator_state=new_discriminator_state,
      policy_params=state.policy_params,  # Not modified.
      key=key,
      steps=state.steps + 1,
  )
  return new_state, loss


class AILSample(NamedTuple):
  discriminator_sample: types.Transition
  direct_sample: reverb.ReplaySample
  demonstration_sample: types.Transition


class AILLearner(acme.Learner):
  """AIL learner."""

  def __init__(
      self,
      counter: counting.Counter,
      direct_rl_learner_factory: Callable[[Iterator[reverb.ReplaySample]],
                                          acme.Learner],
      loss_fn: losses.Loss,
      iterator: Iterator[AILSample],
      discriminator_optimizer: optax.GradientTransformation,
      ail_network: ail_networks.AILNetworks,
      discriminator_key: networks_lib.PRNGKey,
      is_sequence_based: bool,
      num_sgd_steps_per_step: int = 1,
      policy_variable_name: Optional[str] = None,
      logger: Optional[loggers.Logger] = None):
    """AIL Learner.

    Args:
      counter: Counter.
      direct_rl_learner_factory: Function that creates the direct RL learner
        when passed a replay sample iterator.
      loss_fn: Discriminator loss.
      iterator: Iterator that returns AILSamples.
      discriminator_optimizer: Discriminator optax optimizer.
      ail_network: AIL networks.
      discriminator_key: RNG key.
      is_sequence_based: If True, a direct rl algorithm is using SequenceAdder
        data format. Otherwise the learner assumes that the direct rl algorithm
        is using NStepTransitionAdder.
      num_sgd_steps_per_step: Number of discriminator gradient updates per step.
      policy_variable_name: The name of the policy variable to retrieve
        direct_rl policy parameters.
      logger: Logger.
    """
    self._is_sequence_based = is_sequence_based

    state_key, networks_key = jax.random.split(discriminator_key)

    # Generator expression that works the same as an iterator.
    # https://pymbook.readthedocs.io/en/latest/igd.html#generator-expressions
    iterator, direct_rl_iterator = itertools.tee(iterator)
    direct_rl_iterator = (
        self._process_sample(sample.direct_sample)
        for sample in direct_rl_iterator)
    self._direct_rl_learner = direct_rl_learner_factory(direct_rl_iterator)

    self._iterator = iterator

    if policy_variable_name is not None:

      def get_policy_params():
        return self._direct_rl_learner.get_variables([policy_variable_name])[0]

      self._get_policy_params = get_policy_params

    else:
      self._get_policy_params = lambda: None

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray)

    # Use the JIT compiler.
    self._update_step = functools.partial(
        ail_update_step,
        optimizer=discriminator_optimizer,
        ail_network=ail_network,
        loss_fn=loss_fn)
    self._update_step = utils.process_multiple_batches(self._update_step,
                                                       num_sgd_steps_per_step)
    self._update_step = jax.jit(self._update_step)

    discriminator_params, discriminator_state = (
        ail_network.discriminator_network.init(networks_key))
    self._state = DiscriminatorTrainingState(
        optimizer_state=discriminator_optimizer.init(discriminator_params),
        discriminator_params=discriminator_params,
        discriminator_state=discriminator_state,
        policy_params=self._get_policy_params(),
        key=state_key,
        steps=0,
    )

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

    self._get_reward = jax.jit(
        functools.partial(
            ail_networks.compute_ail_reward, networks=ail_network))

  def _process_sample(self, sample: reverb.ReplaySample) -> reverb.ReplaySample:
    """Updates the reward of the replay sample.

    Args:
      sample: Replay sample to update the reward to.

    Returns:
      The replay sample with an updated reward.
    """
    transitions = reverb_utils.replay_sample_to_sars_transition(
        sample, is_sequence=self._is_sequence_based)
    rewards = self._get_reward(self._state.discriminator_params,
                               self._state.discriminator_state,
                               self._state.policy_params, transitions)

    return sample._replace(data=sample.data._replace(reward=rewards))

  def step(self):
    sample = next(self._iterator)
    rb_transitions = sample.discriminator_sample
    demo_transitions = sample.demonstration_sample

    if demo_transitions.reward.shape != rb_transitions.reward.shape:
      raise ValueError(
          'Different shapes for demo transitions and rb_transitions: '
          f'{demo_transitions.reward.shape} != {rb_transitions.reward.shape}')

    # Update the parameters of the policy before doing a gradient step.
    state = self._state._replace(policy_params=self._get_policy_params())
    self._state, metrics = self._update_step(state,
                                             (demo_transitions, rb_transitions))

    # The order is important for AIRL.
    # In AIRL, the discriminator update depends on the logpi of the direct rl
    # policy.
    # When updating the discriminator, we want the logpi for which the
    # transitions were made with and not an updated one.
    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    self._direct_rl_learner.step()

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names: List[str]) -> List[Any]:
    rewarder_dict = {'discriminator': self._state.discriminator_params}

    learner_names = [name for name in names if name not in rewarder_dict]
    learner_dict = {}
    if learner_names:
      learner_dict = dict(
          zip(learner_names,
              self._direct_rl_learner.get_variables(learner_names)))

    variables = [
        rewarder_dict.get(name, learner_dict.get(name, None)) for name in names
    ]
    return variables

  def save(self) -> TrainingState:
    return TrainingState(
        rewarder_state=self._state,
        learner_state=self._direct_rl_learner.save())

  def restore(self, state: TrainingState):
    self._state = state.rewarder_state
    self._direct_rl_learner.restore(state.learner_state)
