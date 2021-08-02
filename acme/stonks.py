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

# print("temporarily force jax to use CPU for debugging")
# jax.config.update('jax_platform_name', "cpu")

### PARAMETERS:

config = dqn_config.DQNConfig()
  # batch_size=10,
  # samples_per_insert=2,
  # min_replay_size=10) # all the default values should be ok, some overrides to speed up testing

MIN_OBSERVATIONS = max(config.batch_size, config.min_replay_size)

print("MIN OBS:", MIN_OBSERVATIONS)

NUM_STEPS_ACTOR = 20 # taken from agent_test


### REVERB AND ENV

environment = fakes.DiscreteEnvironment(
    num_actions=5,
    num_observations=10,
    obs_shape=(10, 5),
    obs_dtype=np.float32,
    episode_length=10)
spec = specs.make_environment_spec(environment)

#### ACTORS AND LEARNERS

# @ray.remote
@ray.remote(resources={"tpu": 1})
class ActorRay():
  def __init__(self, config, address, learner, environment, storage, verbose=False):
    self.verbose = verbose

    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))

    client = reverb.Client(address)
    adder = adders.NStepTransitionAdder(client, config.n_step, config.discount)

    def create_policy():
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
      return policy
    policy = create_policy()

    self._variable_wrapper = VariableSourceRayWrapper(learner)
    self._variable_client = variable_utils.VariableClient(self._variable_wrapper, '')

    self._actor = actors.FeedForwardActor(
      policy=policy,
      random_key=key_actor,
      variable_client=self._variable_client, # need to write a custom wrapper around learner so it calls .remote
      adder=adder)
      # backend="tpu_driver")

    class Printer():
      def write(self, s):
        print(s)

    self._counter = counting.Counter()
    self._logger = Printer()
    # for some reason, default logger is broken
    # TODO: fix this
    # self._logger = loggers.make_default_logger("loop")

    self._environment = environment
    self._should_update = True
    self.storage = storage

    print("actor instantiated")


  @staticmethod
  def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)

  # only used to "initialize" params
  def get_params(self):
    if self.verbose: print("getting params")
    data = self._variable_client.params
    if self.verbose: print("params gotten!")
    return data

  def run(self):
    # need to make this just run ex infinita - perhaps just keep calling run episode myself? lol
    if self.verbose: print("started actor run", jnp.ones(3).device_buffer.device())
    steps = 0
    while not ray.get(self.storage.get_info.remote("terminate")):
      result = self.run_episode()
      steps += result["episode_length"]
      if steps == 0: self._logger.write(result)
      self.storage.set_info.remote({
        "steps": steps
        })

    print(f"terminate received, terminating at {steps} steps")

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(self._generate_zeros_from_spec,
                                        self._environment.reward_spec())
    print("flag 1")
    timestep = self._environment.reset()
    print("flag 1.5")
    # Make the first observation.
    self._actor.observe_first(timestep)
    print("flag 2")
    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      print("flag 2.25")
      action = self._actor.select_action(timestep.observation)
      print("flag 2.5")
      timestep = self._environment.step(action)

      # Have the agent observe the timestep and let the actor update itself.
      self._actor.observe(action, next_timestep=timestep)
      print("flag 3")
      if self._should_update:
        self._actor.update()

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd,
                                          episode_return,
                                          timestep.reward)
    print("flag 4")
    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
    }
    result.update(counts)
    return result

@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """
    def __init__(self):
      self.current_checkpoint = {}

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

# @ray.remote
@ray.remote(resources={"tpu": 1})
class LearnerRay():
  def __init__(self, config, address, storage, verbose=False):
    self.verbose = verbose

    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))

    self.config = config
    self.address = address
    self.storage = storage
    if self.verbose: print("learner addr", address)

    def create_network():
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
      return network

    network = create_network()

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
      network=network,
      random_key=key_learner,
      optimizer=optimizer,
      discount=config.discount,
      importance_sampling_exponent=config.importance_sampling_exponent,
      target_update_period=config.target_update_period,
      iterator=data_iterator,
      replay_client=client
    )
    if self.verbose: print("learner instantiated")


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

  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
    """This has to be called by a wrapper which uses the .remote postfix."""
    return self.learner.get_variables(names)

  def run(self):
    # we just keep count of the number of steps it's trained on
    step_count = 0

    while ray.get(self.storage.get_info.remote("steps")) < MIN_OBSERVATIONS:
      if self.verbose: print(f"sleeping", jnp.ones(3).device_buffer.device())
      time.sleep(1)

    if self.verbose: print("IT FINALLY ESCAPED THANK OUR LORD AND SAVIOUR")

    while step_count < 10:
      num_steps = self._calculate_num_learner_steps(
        num_observations=ray.get(self.storage.get_info.remote("steps")),
        min_observations=MIN_OBSERVATIONS,
        observations_per_step=self.config.batch_size / self.config.samples_per_insert,
        )

      for _ in range(num_steps):
        self.learner.step()

      step_count += num_steps



    print(f"learning complete ({step_count})...terminating self-play")
    self.storage.set_info.remote({
      "terminate": True
      })


class VariableSourceRayWrapper():
  def __init__(self, source):
    self.source = source

  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
    return ray.get(self.source.get_variables.remote(names))

if __name__ == "__main__":
  ray.init(address="auto")

  storage = SharedStorage.remote()
  storage.set_info.remote({
    "steps": 0,
    "terminate": False
    })

  # note that we use `reverb_replay.address` which isn't supported by acme 
  # out of the box, so we'll either have to use something else or patch
  # the source
  reverb_replay = replay.make_reverb_prioritized_nstep_replay(
      environment_spec=spec,
      n_step=config.n_step,
      batch_size=config.batch_size,
      max_replay_size=config.max_replay_size,
      min_replay_size=config.min_replay_size,
      priority_exponent=config.priority_exponent,
      discount=config.discount,
  )

  learner = LearnerRay.options().remote(config, reverb_replay.address, storage, verbose=True)
  # variable_wrapper = VariableSourceRayWrapper(learner)
  actor = ActorRay.options().remote(config, reverb_replay.address, learner, environment, storage, verbose=True)

  # we need to do this because you need to make sure the learner is initialized
  # before the actor can start self-play (it retrieves the params from learner)
  ray.get(actor.get_params.remote())

  actor.run.remote()
  learner.run.remote()

  while not ray.get(storage.get_info.remote("terminate")):
    time.sleep(1)


