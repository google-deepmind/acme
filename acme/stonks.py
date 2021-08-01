### IMPORTS
import time

import acme
from acme import specs
from acme import datasets
from acme.jax import utils
from acme.jax import variable_utils
from acme.jax import networks as networks_lib
from acme.agents import replay
from acme.agents.jax import actors
from acme.agents.jax.dqn import learning
from acme.agents.jax.dqn import config as dqn_config
from acme.testing import fakes

from acme.adders import reverb as adders
from typing import Generic, List, Optional, Sequence, TypeVar
from acme import types


import ray
import jax
import jax.numpy as jnp
import rlax
import optax
import reverb
import numpy as np
import haiku as hk

import operator
import tree

from acme.utils import counting
from acme.utils import loggers



### PARAMETERS:

config = dqn_config.DQNConfig(
  batch_size=10,
  samples_per_insert=2,
  min_replay_size=10) # all the default values should be ok, some overrides to speed up testing

MIN_OBSERVATIONS = max(config.batch_size, config.min_replay_size)
NUM_STEPS_ACTOR = 20 # taken from agent_test

# key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))



### REVERB AND ENV


environment = fakes.DiscreteEnvironment(
    num_actions=5,
    num_observations=10,
    obs_shape=(10, 5),
    obs_dtype=np.float32,
    episode_length=10)
spec = specs.make_environment_spec(environment)


reverb_replay = replay.make_reverb_prioritized_nstep_replay(
    environment_spec=spec,
    n_step=config.n_step,
    batch_size=config.batch_size,
    max_replay_size=config.max_replay_size,
    min_replay_size=config.min_replay_size,
    priority_exponent=config.priority_exponent,
    discount=config.discount,
)

# address = reverb_replay.address


### NETWORK

def network(x):
  model = hk.Sequential([
      hk.Flatten(),
      hk.nets.MLP([50, 50, spec.actions.num_values])
  ])
  return model(x)

# Make network purely functional
network_hk = hk.without_apply_rng(hk.transform(network, apply_rng=True))
dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

network = networks_lib.FeedForwardNetwork(
  init=lambda rng: network_hk.init(rng, dummy_obs),
  apply=network_hk.apply)

def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)



# we have should_update=True so that the actor updates its variables
# this is to start populating the replay buffer, aka self-play

@ray.remote
class StonksActor():
  def __init__(self, config, address, variable_wrapper, environment):
    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))

    client = reverb.Client(address)
    adder = adders.NStepTransitionAdder(client, config.n_step, config.discount)


    def network(x):
      model = hk.Sequential([
          hk.Flatten(),
          hk.nets.MLP([50, 50, spec.actions.num_values])
      ])
      return model(x)

    # Make network purely functional
    network_hk = hk.without_apply_rng(hk.transform(network, apply_rng=True))
    dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

    network = networks_lib.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, dummy_obs),
      apply=network_hk.apply)


    def policy(params: networks_lib.Params, key: jnp.ndarray,
               observation: jnp.ndarray) -> jnp.ndarray:
      action_values = network.apply(params, observation) # how will this work when they're on different devices?
      return rlax.epsilon_greedy(config.epsilon).sample(key, action_values)

    self._actor = actors.FeedForwardActor(
      policy=policy,
      random_key=key_actor,
      variable_client=variable_utils.VariableClient(variable_wrapper, ''), # need to write a custom wrapper around learner so it calls .remote
      adder=adder)


    self._counter = counting.Counter()
    self._logger = loggers.make_default_logger("loop")

    self._environment = environment
    self._should_update = True

    class Printer():
      def write(self, s):
        print(s)
    logger = Printer()


    self.loop = acme.EnvironmentLoop(environment, self._actor, should_update=True, logger=logger)
    # print("actor instantiated")
    # print("running environment loop")
    # loop.run(num_steps=NUM_STEPS_ACTOR) # we'll probably want a custom environment loop which reads a sharedstorage to decide whether to terminate

  def run(self):
    self.loop.run(num_episodes=100)
  # def run_episode(self) -> loggers.LoggingData:
  #   """Run one episode.

  #   Each episode is a loop which interacts first with the environment to get an
  #   observation and then give that observation to the agent in order to retrieve
  #   an action.

  #   Returns:
  #     An instance of `loggers.LoggingData`.
  #   """
  #   # Reset any counts and start the environment.
  #   start_time = time.time()
  #   episode_steps = 0

  #   # For evaluation, this keeps track of the total undiscounted reward
  #   # accumulated during the episode.
  #   episode_return = tree.map_structure(_generate_zeros_from_spec,
  #                                       self._environment.reward_spec())
  #   timestep = self._environment.reset()

  #   # Make the first observation.
  #   self._actor.observe_first(timestep)

  #   # Run an episode.
  #   while not timestep.last():
  #     # Generate an action from the agent's policy and step the environment.
  #     action = self._actor.select_action(timestep.observation)
  #     timestep = self._environment.step(action)

  #     # Have the agent observe the timestep and let the actor update itself.
  #     self._actor.observe(action, next_timestep=timestep)
  #     if self._should_update:
  #       self._actor.update()

  #     # Book-keeping.
  #     episode_steps += 1

  #     # Equivalent to: episode_return += timestep.reward
  #     # We capture the return value because if timestep.reward is a JAX
  #     # DeviceArray, episode_return will not be mutated in-place. (In all other
  #     # cases, the returned episode_return will be the same object as the
  #     # argument episode_return.)
  #     episode_return = tree.map_structure(operator.iadd,
  #                                         episode_return,
  #                                         timestep.reward)

  #   # Record counts.
  #   counts = self._counter.increment(episodes=1, steps=episode_steps)

  #   # Collect the results and combine with counts.
  #   steps_per_second = episode_steps / (time.time() - start_time)
  #   result = {
  #       'episode_length': episode_steps,
  #       'episode_return': episode_return,
  #       'steps_per_second': steps_per_second,
  #   }
  #   result.update(counts)
  #   return result

  # def run(self,
  #         num_episodes: Optional[int] = None,
  #         num_steps: Optional[int] = None):
  #   """Perform the run loop.

  #   Run the environment loop either for `num_episodes` episodes or for at
  #   least `num_steps` steps (the last episode is always run until completion,
  #   so the total number of steps may be slightly more than `num_steps`).
  #   At least one of these two arguments has to be None.

  #   Upon termination of an episode a new episode will be started. If the number
  #   of episodes and the number of steps are not given then this will interact
  #   with the environment infinitely.

  #   Args:
  #     num_episodes: number of episodes to run the loop for.
  #     num_steps: minimal number of steps to run the loop for.

  #   Raises:
  #     ValueError: If both 'num_episodes' and 'num_steps' are not None.
  #   """

  #   if not (num_episodes is None or num_steps is None):
  #     raise ValueError('Either "num_episodes" or "num_steps" should be None.')

  #   def should_terminate(episode_count: int, step_count: int) -> bool:
  #     return ((num_episodes is not None and episode_count >= num_episodes) or
  #             (num_steps is not None and step_count >= num_steps))

  #   episode_count, step_count = 0, 0
  #   while not should_terminate(episode_count, step_count):
  #     result = self.run_episode()
  #     episode_count += 1
  #     step_count += result['episode_length']
  #     # Log the given results.
  #     print(result)
  #     self._logger.write(result)


class VariableSourceRayWrapper():
  def __init__(self, source):
    self.source = source

  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
    return ray.get(self.source.get_variables.remote(names))


@ray.remote
class StonksLearner():
  def __init__(self, config, address):
    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))

    self.config = config
    self.address = address

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(config.learning_rate),
    )

    client = reverb.Client(address)

    data_iterator = datasets.make_reverb_dataset(
        table="priority_table",
        server_address=address,
        batch_size=config.batch_size,
        prefetch_size=4,
    ).as_numpy_iterator()

    self.learner = learning.DQNLearner(
      network=network,# let's try having the same network
      random_key=key_learner,
      optimizer=optimizer,
      discount=config.discount,
      importance_sampling_exponent=config.importance_sampling_exponent,
      target_update_period=config.target_update_period,
      iterator=data_iterator,
      replay_client=client
    )
    print("learner instantiated")


  @staticmethod
  def _calculate_num_learner_steps(num_observations: int, min_observations: int, observations_per_step: float) -> int:
    """Calculates the number of learner steps to do at step=num_observations."""
    n = num_observations - min_observations
    if n < 0:
      # Do not do any learner steps until you have seen min_observations.
      return 0
    if observations_per_step > 1:
      # One batch every 1/obs_per_step observations, otherwise zero.
      return int(n % int(observations_per_step) == 0)
    else:
      # Always return 1/obs_per_step batches every observation.
      return int(1 / observations_per_step)

  def finished(self):
    return True

  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
      return self.learner.get_variables(names)

  def run(self):
    client = reverb.Client(self.address)
    # we just keep count of the number of steps it's trained on
    step_count = 0

    while step_count < 10:
      num_transitions = client.server_info()['priority_table'].num_episodes # should be ok?

      if num_transitions < MIN_OBSERVATIONS:
        print("sleeping")
        time.sleep(1)
        continue

      num_steps = self._calculate_num_learner_steps(
        num_observations=num_transitions,
        min_observations=MIN_OBSERVATIONS,
        observations_per_step=self.config.batch_size / self.config.samples_per_insert,
        )

      for _ in range(num_steps):
        self.learner.step()

      step_count += num_steps

if __name__ == "__main__":
  ray.init()

  learner = StonksLearner.remote(config, reverb_replay.address)

  variable_wrapper = VariableSourceRayWrapper(learner)

  ray.get(learner.finished.remote())
  print("learner should be done first! then only actor")

  actor = StonksActor.remote(config, reverb_replay.address, variable_wrapper, environment)
  # print(ray.get(actor.run.remote(num_episodes=10)))

  # logger = loggers.make_default_logger("loop")

  # num_episodes = 0
  # result = ray.get(actor.run.remote(num_episodes=100))
  result = ray.get(actor.run.remote())

  # while num_episodes < 100:
  #   result = ray.get(actor.run_episode.remote())
  #   print(f"episode {num_episodes} finished:", result)
  #   num_episodes += 1
  #   logger.write(result)



