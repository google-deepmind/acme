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

"""D4PG Builder."""
from typing import Iterator, List, Optional

import jax
import optax
import reverb
import tensorflow as tf
import tree
from reverb import rate_limiters
from reverb import structured_writer as sw

import acme
from acme import adders, core, specs, types
from acme.adders import reverb as adders_reverb
from acme.adders.reverb import base as reverb_base
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors, builders
from acme.agents.jax.d4pg import config as d4pg_config
from acme.agents.jax.d4pg import learning
from acme.agents.jax.d4pg import networks as d4pg_networks
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils, variable_utils
from acme.utils import counting, loggers


def _make_adder_config(
    step_spec: reverb_base.Step, n_step: int, table: str
) -> List[sw.Config]:
    return adders_reverb.create_n_step_transition_config(
        step_spec=step_spec, n_step=n_step, table=table
    )


def _as_n_step_transition(
    flat_trajectory: reverb.ReplaySample, agent_discount: float
) -> reverb.ReplaySample:
    """Compute discounted return and total discount for N-step transitions.

  For N greater than 1, transitions are of the form:

          (s_t, a_t, r_{t:t+n}, r_{t:t+n}, s_{t+N}, e_t),

  where:

      s_t = State (observation) at time t.
      a_t = Action taken from state s_t.
      g = the additional discount, used by the agent to discount future returns.
      r_{t:t+n} = A vector of N-step rewards: [r_t r_{t+1} ... r_{t+n}]
      d_{t:t+n} = A vector of N-step environment: [d_t d_{t+1} ... d_{t+n}]
        For most environments d_i is 1 for all steps except the last,
        i.e. it is the episode termination signal.
      s_{t+n}: The "arrival" state, i.e. the state at time t+n.
      e_t [Optional]: A nested structure of any 'extras' the user wishes to add.

  As such postprocessing is necessary to calculate the N-Step discounted return
  and the total discount as follows:

          (s_t, a_t, R_{t:t+n}, D_{t:t+n}, s_{t+N}, e_t),

    where:

      R_{t:t+n} = N-step discounted return, i.e. accumulated over N rewards:
            R_{t:t+n} := r_t + g * d_t * r_{t+1} + ...
                            + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}.
      D_{t:t+n}: N-step product of agent discounts g_i and environment
        "discounts" d_i.
            D_{t:t+n} := g^{n-1} * d_{t} * ... * d_{t+n-1},

  Args:
    flat_trajectory: An trajectory with n-step rewards and discounts to be
      process.
    agent_discount: An additional discount factor used by the agent to discount
      futrue returns.

  Returns:
    A reverb.ReplaySample with computed discounted return and total discount.
  """
    trajectory = flat_trajectory.data

    def compute_discount_and_reward(
        state: types.NestedTensor, discount_and_reward: types.NestedTensor
    ) -> types.NestedTensor:
        compounded_discount, discounted_reward = state
        return (
            agent_discount * discount_and_reward[0] * compounded_discount,
            discounted_reward + discount_and_reward[1] * compounded_discount,
        )

    initializer = (tf.constant(1, dtype=tf.float32), tf.constant(0, dtype=tf.float32))
    elems = tf.stack((trajectory.discount, trajectory.reward), axis=-1)
    total_discount, n_step_return = tf.scan(
        compute_discount_and_reward, elems, initializer, reverse=True
    )
    return reverb.ReplaySample(
        info=flat_trajectory.info,
        data=types.Transition(
            observation=tree.map_structure(lambda x: x[0], trajectory.observation),
            action=tree.map_structure(lambda x: x[0], trajectory.action),
            reward=n_step_return[0],
            discount=total_discount[0],
            next_observation=tree.map_structure(
                lambda x: x[-1], trajectory.observation
            ),
            extras=tree.map_structure(lambda x: x[0], trajectory.extras),
        ),
    )


class D4PGBuilder(
    builders.ActorLearnerBuilder[
        d4pg_networks.D4PGNetworks, actor_core_lib.ActorCore, reverb.ReplaySample
    ]
):
    """D4PG Builder."""

    def __init__(
        self, config: d4pg_config.D4PGConfig,
    ):
        """Creates a D4PG learner, a behavior policy and an eval actor.

    Args:
      config: a config with D4PG hps
    """
        self._config = config

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: d4pg_networks.D4PGNetworks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        del environment_spec, replay_client

        policy_optimizer = optax.adam(self._config.learning_rate)
        critic_optimizer = optax.adam(self._config.learning_rate)

        if self._config.clipping:
            policy_optimizer = optax.chain(
                optax.clip_by_global_norm(40.0), policy_optimizer
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(40.0), critic_optimizer
            )

        # The learner updates the parameters (and initializes them).
        return learning.D4PGLearner(
            policy_network=networks.policy_network,
            critic_network=networks.critic_network,
            random_key=random_key,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=self._config.clipping,
            discount=self._config.discount,
            target_update_period=self._config.target_update_period,
            iterator=dataset,
            counter=counter,
            logger=logger_fn("learner"),
            num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        )

    def make_replay_tables(
        self, environment_spec: specs.EnvironmentSpec, policy: actor_core_lib.ActorCore,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        dummy_actor_state = policy.init(jax.random.PRNGKey(0))
        extras_spec = policy.get_extras(dummy_actor_state)
        step_spec = adders_reverb.create_step_spec(
            environment_spec=environment_spec, extras_spec=extras_spec
        )

        # Create the rate limiter.
        if self._config.samples_per_insert:
            samples_per_insert_tolerance = (
                self._config.samples_per_insert_tolerance_rate
                * self._config.samples_per_insert
            )
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer,
            )
        else:
            limiter = rate_limiters.MinSize(self._config.min_replay_size)
        return [
            reverb.Table(
                name=self._config.replay_table_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self._config.max_replay_size,
                rate_limiter=limiter,
                signature=sw.infer_signature(
                    configs=_make_adder_config(
                        step_spec, self._config.n_step, self._config.replay_table_name
                    ),
                    step_spec=step_spec,
                ),
            )
        ]

    def make_dataset_iterator(
        self, replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""

        def postprocess(flat_trajectory: reverb.ReplaySample) -> reverb.ReplaySample:
            return _as_n_step_transition(flat_trajectory, self._config.discount)

        batch_size_per_device = self._config.batch_size // jax.device_count()

        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=batch_size_per_device * self._config.num_sgd_steps_per_step,
            prefetch_size=self._config.prefetch_size,
            postprocess=postprocess,
        )
        return utils.multi_device_put(dataset.as_numpy_iterator(), jax.local_devices())

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec],
        policy: Optional[actor_core_lib.ActorCore],
    ) -> Optional[adders.Adder]:
        """Create an adder which records data generated by the actor/environment."""
        if environment_spec is None or policy is None:
            raise ValueError("`environment_spec` and `policy` cannot be None.")
        dummy_actor_state = policy.init(jax.random.PRNGKey(0))
        extras_spec = policy.get_extras(dummy_actor_state)
        step_spec = adders_reverb.create_step_spec(
            environment_spec=environment_spec, extras_spec=extras_spec
        )
        return adders_reverb.StructuredAdder(
            client=replay_client,
            max_in_flight_items=5,
            configs=_make_adder_config(
                step_spec, self._config.n_step, self._config.replay_table_name
            ),
            step_spec=step_spec,
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: actor_core_lib.ActorCore,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> acme.Actor:
        del environment_spec
        assert variable_source is not None
        # Inference happens on CPU, so it's better to move variables there too.
        variable_client = variable_utils.VariableClient(
            variable_source, "policy", device="cpu"
        )
        return actors.GenericActor(
            policy, random_key, variable_client, adder, backend="cpu"
        )

    def make_policy(
        self,
        networks: d4pg_networks.D4PGNetworks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> actor_core_lib.ActorCore:
        """Create the policy."""
        del environment_spec
        if evaluation:
            policy = d4pg_networks.get_default_eval_policy(networks)
        else:
            policy = d4pg_networks.get_default_behavior_policy(networks, self._config)

        return actor_core_lib.batched_feed_forward_to_actor_core(policy)
