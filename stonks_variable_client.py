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
import uuid


### PARAMETERS:

config = dqn_config.DQNConfig(
  learning_rate=1e-3,
  samples_per_insert=0.5
)

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

def make_network():
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
  network = make_network()

  def policy(params: networks_lib.Params, key: jnp.ndarray,
             observation: jnp.ndarray) -> jnp.ndarray:
    action_values = network.apply(params, observation) # how will this work when they're on different devices?
    return rlax.epsilon_greedy(config.epsilon).sample(key, action_values)
  return policy


#### ACTORS AND LEARNERS

#@ray.remote(resources={"tpu": 1})
@ray.remote
class ActorRay():
  def __init__(self, config, address, learner, storage, environment_maker, policy_maker, id=None, print_interval=100, verbose=False):
    self.verbose = verbose
    self._should_update = True
    self.storage = storage

    environment = environment_maker()
    self._environment = environment

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

    self._last_ten_episode_info = []
    self.id = id or uuid.uuid1()
    self.print_interval = print_interval # step interval for printing info

    print("actor instantiated")

  @property
  def num_updates(self):
    return self._variable_client._num_updates

  @property
  def info(self):
    info = {
      "id": self.id,
      "num_updates": self.num_updates,
      "avg_statistics": {
        "episode_return": sum([e["episode_return"] for e in self._last_ten_episode_info])/len(self._last_ten_episode_info),
        "episode_length": sum([e["episode_length"] for e in self._last_ten_episode_info])/len(self._last_ten_episode_info),
        "steps_per_second": sum([e["steps_per_second"] for e in self._last_ten_episode_info])/len(self._last_ten_episode_info),
      },
      "total_steps": self._last_ten_episode_info[-1]["steps"],
      "total_episodes": self._last_ten_episode_info[-1]["episodes"],
    }
    return info

  @staticmethod
  def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)

  # only used to "initialize" params
  def get_params(self):
    if self.verbose: print("getting params")
    data = self._variable_client.params
    if self.verbose: print("params gotten!")
    return data

  
  # used to print the current status of the actor, optionally takes a result
  def print_status(self, result=None):
    data = self.info
    if result: data["manual_result"] = result

    print(data)


  def run(self):
    if self.verbose: print(f"started actor {self.id} run", jnp.ones(3).device_buffer.device())
    steps = 0

    while not ray.get(self.storage.get_info.remote("terminate")):
      result = self.run_episode()
      # print(result)

      self._last_ten_episode_info.append(result)
      if len(self._last_ten_episode_info) > 10:
        self._last_ten_episode_info.pop(0)

      steps += result["episode_length"]
      if steps == 0: self._logger.write(result)

      if steps % self.print_interval == 0:
        self.print_status(result)

      # self.storage.set_info.remote({
      #   "steps": steps
      #   })

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
        # Acme usually does `wait=True`, but it turns out that you can get excellent
        # learning even when the actors' weight updates occur asynchronously (and
        # are therefore delayed).
        self._actor.update(wait=False)

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
  def __init__(self, config, address, storage, environment_maker, network_maker, policy_maker, eval_interval=None, verbose=False):
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

    # create an eval learner
    self.environment_maker = environment_maker
    self.eval_interval = eval_interval
    adder = adders.NStepTransitionAdder(self.client, config.n_step, config.discount)
    self._variable_client = variable_utils.VariableClient(self.learner, '')
    policy = policy_maker()
    self.eval_actor = actors.FeedForwardActor(
      policy=policy,
      random_key=key_actor,
      variable_client=self._variable_client, # need to write a custom wrapper around learner so it calls .remote
      adder=adder)

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


  def run(self, total_learning_steps: int = 3e8):
    # we just keep count of the number of steps it's trained on
    step_count = 0

    while self.client.server_info()["priority_table"].current_size < MIN_OBSERVATIONS:
      time.sleep(0.1)

    if self.verbose: print("IT FINALLY ESCAPED THANK OUR LORD AND SAVIOUR")

    while step_count < total_learning_steps:
      num_steps = self._calculate_num_learner_steps(
        num_observations=self.client.server_info()["priority_table"].current_size,
        min_observations=MIN_OBSERVATIONS,
        observations_per_step=self.config.batch_size / self.config.samples_per_insert,
        )

      for _ in range(num_steps):
        self.learner.step()
        step_count += 1
        if self.eval_interval and (step_count % self.eval_interval == 0):
          self.evaluate()

      # step_count += num_steps

    print(f"learning complete ({step_count})...terminating self-play")
    self.storage.set_info.remote({
      "terminate": True
      })

  def evaluate(self, total_eval_episodes = 1, print_info=True):
    print("running eval")
    class ResultStorage():
      def __init__(self):
        self.returns = []

      def write(self, s):
        self.returns.append(s)

    results_logger = ResultStorage()
    eval_env = self.environment_maker()

    # todo: create a custom environment loop for eval so that we can override the `wait` for self.update
    eval_loop = acme.EnvironmentLoop(eval_env, self.eval_actor, logger=results_logger)
    eval_loop.run(num_episodes=total_eval_episodes)

    # calculate stats and return them
    info = {
      "id": "eval",
      "num_updates": self._variable_client._num_updates, # across ALL evals
      "avg_statistics": {
        "episode_return": sum([e["episode_return"] for e in results_logger.returns])/len(results_logger.returns),
        "episode_length": sum([e["episode_length"] for e in results_logger.returns])/len(results_logger.returns),
        "steps_per_second": sum([e["steps_per_second"] for e in results_logger.returns])/len(results_logger.returns),
      },
      "total_steps": results_logger.returns[-1]["steps"],
      "total_episodes": results_logger.returns[-1]["episodes"],
    }

    if print_info: print(info)

    return info

class VariableSourceRayWrapper():
  def __init__(self, source):
    self.source = source

  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
    return ray.get(self.source.get_variables.remote(names))

  def __getattr__(self, name: str):
    # Expose any other attributes of the underlying environment.
    return getattr(self._environment, name)


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

  learner = LearnerRay.options(max_concurrency=32).remote(config, f"{HEAD_IP}:{HEAD_PORT}", storage, make_environment, make_network, make_policy, eval_interval=10, verbose=True)
  actors = [
    ActorRay.options().remote(
      config,
      f"{HEAD_IP}:{HEAD_PORT}",
      learner,
      storage,
      make_environment,
      make_policy,
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
