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
# from acme.testing import fakes

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

import functools
import dm_env
import gym
from acme import wrappers
import bsuite
import copy


### PARAMETERS:

config = dqn_config.DQNConfig()

MIN_OBSERVATIONS = max(config.batch_size, config.min_replay_size)
NUM_STEPS_ACTOR = 32 # taken from agent_test
HEAD_IP = "localhost"
HEAD_PORT = 8000


### REVERB AND ENV

def make_environment(evaluation: bool = False, level: str = 'BreakoutNoFrameskip-v4') -> dm_env.Environment:
  
  env = gym.make(level, full_action_space=True)
  max_episode_len = 108_000 if evaluation else 50_000

  return wrappers.wrap_all(env, [
      wrappers.GymAtariAdapter,
      functools.partial(
          wrappers.AtariWrapper,
          to_float=True,
          max_episode_len=max_episode_len,
          zero_discount_on_life_loss=True,
      ),
      wrappers.SinglePrecisionWrapper,
  ])

demo_env = make_environment()
spec = specs.make_environment_spec(demo_env)


def create_network():
  def network(x):
    model = hk.Sequential([
        networks_lib.AtariTorso(),
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

def make_policy():
  network = create_network()

  def policy(params: networks_lib.Params, key: jnp.ndarray,
             observation: jnp.ndarray) -> jnp.ndarray:
    action_values = network.apply(params, observation) # how will this work when they're on different devices?
    return rlax.epsilon_greedy(config.epsilon).sample(key, action_values)
  return policy


#### ACTORS AND LEARNERS

#@ray.remote(resources={"tpu": 1})
@ray.remote
class ActorRay():
  def __init__(self, config, address, learner, storage, environment_maker, policy_maker, verbose=False):
    environment = environment_maker()
    self.verbose = verbose

    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))

    client = reverb.Client(address)
    adder = adders.NStepTransitionAdder(client, config.n_step, config.discount)

    policy = policy_maker()

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
    if self.verbose: print("started actor run", jnp.ones(3).device_buffer.device())
    steps = 0

    while not ray.get(self.storage.get_info.remote("terminate")):
      result = self.run_episode()
      print(result)

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
    timestep = self._environment.reset()
    # Make the first observation.
    self._actor.observe_first(timestep)
    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      action = self._actor.select_action(timestep.observation)
      timestep = self._environment.step(action)

      # Have the agent observe the timestep and let the actor update itself.
      self._actor.observe(action, next_timestep=timestep)

      if self._should_update:
        self._actor.update(wait=True)

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


@ray.remote(num_cpus=32, resources={"tpu": 1})
class LearnerRay():
  def __init__(self, config, address, storage, network_maker, verbose=False):
    self.verbose = verbose

    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))

    self.config = config
    self.address = address
    self.storage = storage
    if self.verbose: print("learner addr", address)

    network = network_maker()

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(config.learning_rate),
    )

    self.client = reverb.Client(address)

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
      replay_client=self.client
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

    while self.client.server_info()["priority_table"].current_size < MIN_OBSERVATIONS:
      time.sleep(0.1)

    if self.verbose: print("IT FINALLY ESCAPED THANK OUR LORD AND SAVIOUR")

    # save_states = []

    while step_count < 3e8:
      num_steps = self._calculate_num_learner_steps(
        num_observations=self.client.server_info()["priority_table"].current_size,
        min_observations=MIN_OBSERVATIONS,
        observations_per_step=self.config.batch_size / self.config.samples_per_insert,
        )

      # if num_steps != 0:
        # save_states.append(self.learner.save().params)
        # print(f"stepping learner {num_steps}")

      # if step_count > 50:
      #   if str(save_states[0]) == str(save_states[1]):
      #     print("you fuckup")
      #   else:
      #     print("all ok")

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

  learner = LearnerRay.options(max_concurrency=32).remote(config, f"{HEAD_IP}:{HEAD_PORT}", storage, create_network, verbose=True)
  actors = [
    ActorRay.options().remote(
      config,
      f"{HEAD_IP}:{HEAD_PORT}",
      learner,
      storage,
      make_environment,
      make_policy
      verbose=True
    ) for actor_id in range(NUM_STEPS_ACTOR)
  ]

  # we need to do this because you need to make sure the learner is initialized
  # before the actor can start self-play (it retrieves the params from learner)
  # ray.get(actor.get_params.remote())
  for a in actors:
    a.run.remote()
  learner.run.remote()

  while not ray.get(storage.get_info.remote("terminate")):
    time.sleep(1)
