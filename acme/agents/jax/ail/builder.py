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

"""Adversarial Imitation Learning (AIL) Builder."""

import functools
import itertools
from typing import Callable, Generic, Iterator, List, Optional, Tuple

from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.adders import reverb as adders_reverb
from acme.agents.jax import builders
from acme.agents.jax.ail import config as ail_config
from acme.agents.jax.ail import learning
from acme.agents.jax.ail import losses
from acme.agents.jax.ail import networks as ail_networks
from acme.datasets import reverb as datasets
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.imitation_learning_types import DirectPolicyNetwork
from acme.utils import counting
from acme.utils import loggers
from acme.utils import reverb_utils
import jax
import numpy as np
import optax
import reverb
from reverb import rate_limiters
import tree


def _split_transitions(
    transitions: types.Transition,
    index: int) -> Tuple[types.Transition, types.Transition]:
  """Splits the given transition on the first axis at the given index.

  Args:
    transitions: Transitions to split.
    index: Spliting index.

  Returns:
    A pair of transitions, the first containing elements before the index
    (exclusive) and the second after the index (inclusive)
  """
  return (tree.map_structure(lambda x: x[:index], transitions),
          tree.map_structure(lambda x: x[index:], transitions))


def _rebatch(iterator: Iterator[types.Transition],
             batch_size: int) -> Iterator[types.Transition]:
  """Rebatch the itererator with the given batch size.

  Args:
    iterator: Iterator to rebatch.
    batch_size: New batch size.

  Yields:
    Transitions with the new batch size.
  """
  data = next(iterator)
  while True:
    while len(data.reward) < batch_size:
      # Ensure we can get enough demonstrations.
      next_data = next(iterator)
      data = tree.map_structure(lambda *args: np.concatenate(list(args)), data,
                                next_data)
    output, data = _split_transitions(data, batch_size)
    yield output


def _mix_arrays(
    replay: np.ndarray,
    demo: np.ndarray,
    index: int,
    seed: int) -> np.ndarray:
  """Mixes `replay` and `demo`.

  Args:
    replay: Replay data to mix. Only index element will be selected.
    demo: Demonstration data to mix.
    index: Amount of replay data we should include.
    seed: RNG seed.

  Returns:
    An array with replay elements up to 'index' and all the demos.
  """
  # We're throwing away some replay data here. We have to if we want to make
  # sure the output info field is correct.
  output = np.concatenate((replay[:index], demo))
  return np.random.default_rng(seed=seed).permutation(output)


def _generate_samples_with_demonstrations(
    demonstration_iterator: Iterator[types.Transition],
    replay_iterator: Iterator[reverb.ReplaySample],
    policy_to_expert_data_ratio: int,
    batch_size) -> Iterator[reverb.ReplaySample]:
  """Generator which creates the sample having demonstrations in them.

  It takes the demonstrations and replay iterators and generates batches with
  same size as the replay iterator, such that each batches have the ratio of
  policy and expert data specified in policy_to_expert_data_ratio on average.
  There is no constraints on how the demonstrations and replay samples should be
  batched.

  Args:
    demonstration_iterator: Iterator of demonstrations.
    replay_iterator: Replay buffer sample iterator.
    policy_to_expert_data_ratio: Amount of policy transitions for 1 expert
      transitions.
    batch_size: Output batch size, which should match the replay batch size.

  Yields:
    Samples having a mix of demonstrations and policy data. The info will match
    the current replay sample info and the batch size will be the same as the
    replay_iterator data batch size.
  """
  count = 0
  if batch_size % (policy_to_expert_data_ratio + 1) != 0:
    raise ValueError(
        'policy_to_expert_data_ratio + 1 must divide the batch size but '
        f'{batch_size} % {policy_to_expert_data_ratio+1} !=0')
  demo_insertion_size = batch_size // (policy_to_expert_data_ratio + 1)
  policy_insertion_size = batch_size - demo_insertion_size

  demonstration_iterator = _rebatch(demonstration_iterator, demo_insertion_size)
  for sample, demos in zip(replay_iterator, demonstration_iterator):
    output_transitions = tree.map_structure(
        functools.partial(_mix_arrays,
                          index=policy_insertion_size,
                          seed=count),
        sample.data, demos)
    count += 1
    yield reverb.ReplaySample(info=sample.info, data=output_transitions)


class AILBuilder(builders.ActorLearnerBuilder[ail_networks.AILNetworks,
                                              DirectPolicyNetwork,
                                              learning.AILSample],
                 Generic[ail_networks.DirectRLNetworks, DirectPolicyNetwork]):
  """AIL Builder."""

  def __init__(
      self,
      rl_agent: builders.ActorLearnerBuilder[ail_networks.DirectRLNetworks,
                                             DirectPolicyNetwork,
                                             reverb.ReplaySample],
      config: ail_config.AILConfig,
      discriminator_loss: losses.Loss,
      make_demonstrations: Callable[[int], Iterator[types.Transition]],
      logger_fn: Callable[[], loggers.Logger] = lambda: None):
    """Implements a builder for AIL using rl_agent as forward RL algorithm.

    Args:
      rl_agent: The standard RL agent used by AIL to optimize the generator.
      config: a AIL config
      discriminator_loss: The loss function for the discriminator to minimize.
      make_demonstrations: A function that returns an iterator with
        demonstrations to be imitated.
      logger_fn: a logger factory for the learner
    """
    self._rl_agent = rl_agent
    self._config = config
    self._discriminator_loss = discriminator_loss
    self._make_demonstrations = make_demonstrations
    self._logger_fn = logger_fn

  def make_learner(self,
                   random_key: jax_types.PRNGKey,
                   networks: ail_networks.AILNetworks,
                   dataset: Iterator[learning.AILSample],
                   logger_fn: loggers.LoggerFactory,
                   environment_spec: specs.EnvironmentSpec,
                   replay_client: Optional[reverb.Client] = None,
                   counter: Optional[counting.Counter] = None) -> core.Learner:
    counter = counter or counting.Counter()
    direct_rl_counter = counting.Counter(counter, 'direct_rl')
    batch_size_per_learner_step = ail_config.get_per_learner_step_batch_size(
        self._config)

    direct_rl_learner_key, discriminator_key = jax.random.split(random_key)

    direct_rl_learner = functools.partial(
        self._rl_agent.make_learner,
        direct_rl_learner_key,
        networks.direct_rl_networks,
        logger_fn=logger_fn,
        environment_spec=environment_spec,
        replay_client=replay_client,
        counter=direct_rl_counter)

    discriminator_optimizer = (
        self._config.discriminator_optimizer or optax.adam(1e-5))

    return learning.AILLearner(
        counter,
        direct_rl_learner_factory=direct_rl_learner,
        loss_fn=self._discriminator_loss,
        iterator=dataset,
        discriminator_optimizer=discriminator_optimizer,
        ail_network=networks,
        discriminator_key=discriminator_key,
        is_sequence_based=self._config.is_sequence_based,
        num_sgd_steps_per_step=batch_size_per_learner_step //
        self._config.discriminator_batch_size,
        policy_variable_name=self._config.policy_variable_name,
        logger=logger_fn('learner', steps_key=counter.get_steps_key()))

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: DirectPolicyNetwork,
  ) -> List[reverb.Table]:
    replay_tables = self._rl_agent.make_replay_tables(environment_spec, policy)
    if self._config.share_iterator:
      return replay_tables
    replay_tables.append(
        reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=rate_limiters.MinSize(self._config.min_replay_size),
            signature=adders_reverb.NStepTransitionAdder.signature(
                environment_spec)))
    return replay_tables

  # This function does not expose all the iterators used by the learner when
  # share_iterator is False, making further wrapping impossible.
  # TODO(eorsini): Expose all iterators.
  # Currently GAIL uses 3 iterators, instead we can make it use a single
  # iterator and return this one here. The way to achieve this would be:
  # * Create the 3 iterators here.
  # * zip them and return them here.
  # * upzip them in the learner (this step will not be necessary once we move to
  # stateless learners)
  # This should work fine as the 3 iterators are always iterated in parallel
  # (i.e. at every step we call next once on each of them).
  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[learning.AILSample]:
    batch_size_per_learner_step = ail_config.get_per_learner_step_batch_size(
        self._config)

    iterator_demonstration = self._make_demonstrations(
        batch_size_per_learner_step)

    direct_iterator = self._rl_agent.make_dataset_iterator(replay_client)

    if self._config.share_iterator:
      # In order to reuse the iterator return values and not lose a 2x factor on
      # sample efficiency, we need to use itertools.tee().
      discriminator_iterator, direct_iterator = itertools.tee(direct_iterator)
    else:
      discriminator_iterator = datasets.make_reverb_dataset(
          table=self._config.replay_table_name,
          server_address=replay_client.server_address,
          batch_size=ail_config.get_per_learner_step_batch_size(self._config),
          prefetch_size=self._config.prefetch_size).as_numpy_iterator()

    if self._config.policy_to_expert_data_ratio is not None:
      iterator_demonstration, iterator_demonstration2 = itertools.tee(
          iterator_demonstration)
      direct_iterator = _generate_samples_with_demonstrations(
          iterator_demonstration2, direct_iterator,
          self._config.policy_to_expert_data_ratio,
          self._config.direct_rl_batch_size)

    is_sequence_based = self._config.is_sequence_based

    # Don't flatten the discriminator batch if the iterator is not shared.
    process_discriminator_sample = functools.partial(
        reverb_utils.replay_sample_to_sars_transition,
        is_sequence=is_sequence_based and self._config.share_iterator,
        flatten_batch=is_sequence_based and self._config.share_iterator,
        strip_last_transition=is_sequence_based and self._config.share_iterator)

    discriminator_iterator = (
        # Remove the extras to have the same nested structure as demonstrations.
        process_discriminator_sample(sample)._replace(extras=())
        for sample in discriminator_iterator)

    return utils.device_put((learning.AILSample(*sample) for sample in zip(
        discriminator_iterator, direct_iterator, iterator_demonstration)),
                            jax.devices()[0])

  def make_adder(self, replay_client: reverb.Client) -> Optional[adders.Adder]:
    direct_rl_adder = self._rl_agent.make_adder(replay_client)
    if self._config.share_iterator:
      return direct_rl_adder
    ail_adder = adders_reverb.NStepTransitionAdder(
        priority_fns={self._config.replay_table_name: None},
        client=replay_client,
        n_step=1,
        discount=self._config.discount)

    # Some direct rl algorithms (such as PPO), might be passing extra data
    # which we won't be able to process here properly, so we need to ignore them
    return adders.ForkingAdder(
        [adders.IgnoreExtrasAdder(ail_adder), direct_rl_adder])

  def make_actor(
      self,
      random_key: jax_types.PRNGKey,
      policy: DirectPolicyNetwork,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> core.Actor:
    return self._rl_agent.make_actor(random_key, policy, environment_spec,
                                     variable_source, adder)

  def make_policy(self,
                  networks: ail_networks.AILNetworks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> DirectPolicyNetwork:
    return self._rl_agent.make_policy(networks.direct_rl_networks,
                                      environment_spec, evaluation)
