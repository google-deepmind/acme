### IMPORTS
import acme
from acme import specs
from acme.jax import utils
from acme.jax import variable_utils
from acme.jax import networks as networks_lib
from acme.agents import replay
from acme.agents.jax import actors
from acme.agents.jax.dqn import learning
from acme.agents.jax.dqn import config as dqn_config
from acme.testing import fakes

from acme.adders import reverb as adders

import ray
import jax
import jax.numpy as jnp
import rlax
import optax
import reverb
import numpy as np
import haiku as hk



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


### NETWORK

# def network(x):
#   model = hk.Sequential([
#       hk.Flatten(),
#       hk.nets.MLP([50, 50, spec.actions.num_values])
#   ])
#   return model(x)

# # Make network purely functional
# network_hk = hk.without_apply_rng(hk.transform(network, apply_rng=True))
# dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

# network = networks_lib.FeedForwardNetwork(
#   init=lambda rng: network_hk.init(rng, dummy_obs),
#   apply=network_hk.apply)



### LEARNER: CREATE

# def _calculate_num_learner_steps(num_observations: int, min_observations: int, observations_per_step: float) -> int:
#   """Calculates the number of learner steps to do at step=num_observations."""
#   n = num_observations - min_observations
#   if n < 0:
#     # Do not do any learner steps until you have seen min_observations.
#     return 0
#   if observations_per_step > 1:
#     # One batch every 1/obs_per_step observations, otherwise zero.
#     return int(n % int(observations_per_step) == 0)
#   else:
#     # Always return 1/obs_per_step batches every observation.
#     return int(1 / observations_per_step)


# optimizer = optax.chain(
#     optax.clip_by_global_norm(config.max_gradient_norm),
#     optax.adam(config.learning_rate),
# )

# learner = learning.DQNLearner(
#   network=network,# let's try having the same network
#   random_key=key_learner,
#   optimizer=optimizer,
#   discount=config.discount,
#   importance_sampling_exponent=config.importance_sampling_exponent,
#   target_update_period=config.target_update_period,
#   iterator=reverb_replay.data_iterator,
#   replay_client=reverb_replay.client
# )



### ACTOR: CREATE AND START ###



# we have should_update=True so that the actor updates its variables
# this is to start populating the replay buffer, aka self-play

@ray.remote
class StonksActor():
  def __init__(self):
    # import acme
    # from acme import specs
    # from acme.jax import utils
    # from acme.jax import variable_utils
    # from acme.jax import networks as networks_lib
    # from acme.agents import replay
    # from acme.agents.jax import actors
    # from acme.agents.jax.dqn import learning
    # from acme.agents.jax.dqn import config as dqn_config
    # from acme.testing import fakes

    # from acme.adders import reverb as adders

    # import ray
    # import jax
    # import jax.numpy as jnp
    # import rlax
    # import optax
    # import reverb
    # import numpy as np
    # import haiku as hk

    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))

    address = 'localhost:8000'
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

    actor = actors.FeedForwardActor(
      policy=policy,
      random_key=key_actor,
      variable_client=variable_utils.VariableClient(learner, ''), # need to write a custom wrapper around learner so it calls .remote
      adder=adder)

    loop = acme.EnvironmentLoop(environment, actor, should_update=True)
    print("running environment loop")
    loop.run(num_steps=NUM_STEPS_ACTOR) # we'll probably want a custom environment loop which reads a sharedstorage to decide whether to terminate



def run_actor():
  pass


@ray.remote
def run_learner():
  # we just keep count of the number of steps it's trained on
  step_count = 0

  while not should_terminate(episode_count, step_count):
    num_transitions = reverb_replay.client.server_info()['priority_table'].num_episodes # should be ok?

    if num_episodes < MIN_OBSERVATIONS:
      sleep(0.5)
      continue

    num_steps = _calculate_num_learner_steps(
      num_observations=num_transitions,
      min_observations=MIN_OBSERVATIONS,
      observations_per_step=config.batch_size / config.samples_per_insert,
      )

    for _ in range(num_steps):
      learner.step()

if __name__ == "__main__":
  ray.init()

  StonksActor.remote()
  # run_actor.remote()
  run_learner.remote()
