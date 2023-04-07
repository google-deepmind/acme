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

"""JAX multiagent builders."""

from typing import Dict, Iterator, List, Mapping, Optional, Sequence

import jax
import reverb

from acme import adders, core, specs, types
from acme.agents.jax import builders as acme_builders
from acme.agents.jax.multiagent.decentralized import actor
from acme.agents.jax.multiagent.decentralized import (
    factories as decentralized_factories,
)
from acme.agents.jax.multiagent.decentralized import learner_set
from acme.jax import networks as networks_lib
from acme.multiagent import types as ma_types
from acme.multiagent import utils as ma_utils
from acme.utils import counting, iterator_utils, loggers

VARIABLE_SEPARATOR = "-"


class PrefixedVariableSource(core.VariableSource):
    """Wraps a variable source to add a pre-defined prefix to all names."""

    def __init__(self, source: core.VariableSource, prefix: str):
        self._source = source
        self._prefix = prefix

    def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
        return self._source.get_variables([self._prefix + name for name in names])


class DecentralizedMultiAgentBuilder(
    acme_builders.GenericActorLearnerBuilder[
        ma_types.MultiAgentNetworks,
        ma_types.MultiAgentPolicyNetworks,
        ma_types.MultiAgentSample,
    ]
):
    """Builder for decentralized multiagent setup."""

    def __init__(
        self,
        agent_types: Dict[ma_types.AgentID, ma_types.GenericAgent],
        agent_configs: Dict[ma_types.AgentID, ma_types.AgentConfig],
        init_policy_network_fn: Optional[ma_types.InitPolicyNetworkFn] = None,
    ):
        """Initializer.

    Args:
      agent_types: Dict mapping agent IDs to their types.
      agent_configs: Dict mapping agent IDs to their configs.
      init_policy_network_fn: Optional custom policy network initializer
        function.
    """

        self._agent_types = agent_types
        self._agent_configs = agent_configs
        self._builders = decentralized_factories.builder_factory(
            agent_types, agent_configs
        )
        self._num_agents = len(self._builders)
        self._init_policy_network_fn = init_policy_network_fn

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: ma_types.MultiAgentPolicyNetworks,
    ) -> List[reverb.Table]:
        """Returns replay tables for all agents.

    Args:
      environment_spec: the (multiagent) environment spec, which will be
        factorized into single-agent specs for replay table initialization.
      policy: the (multiagent) mapping from agent ID to the corresponding
        agent's policy, used to get the correct extras_spec.
    """
        replay_tables = []
        for agent_id, builder in self._builders.items():
            single_agent_spec = ma_utils.get_agent_spec(environment_spec, agent_id)
            replay_tables += builder.make_replay_tables(
                single_agent_spec, policy[agent_id]
            )
        return replay_tables

    def make_dataset_iterator(
        self, replay_client: reverb.Client
    ) -> Iterator[ma_types.MultiAgentSample]:
        # Zipping stores sub-iterators in the order dictated by
        # self._builders.values(), which are insertion-ordered in Python3.7+.
        # Hence, later unzipping (in make_learner()) and accessing the iterators
        # via the same self._builders.items() dict ordering should be safe.
        return zip(
            *[b.make_dataset_iterator(replay_client) for b in self._builders.values()]
        )

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: ma_types.MultiAgentNetworks,
        dataset: Iterator[ma_types.MultiAgentSample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> learner_set.SynchronousDecentralizedLearnerSet:
        """Returns multiagent learner set.

    Args:
      random_key: random key.
      networks: dict of networks, one per learner. Networks can be heterogeneous
        (i.e., distinct in architecture) across learners.
      dataset: list of iterators over samples from replay, one per learner.
      logger_fn: factory providing loggers used for logging progress.
      environment_spec: the (multiagent) environment spec, which will be
        factorized into single-agent specs for replay table initialization.
      replay_client: replay client that is shared amongst the sub-learners.
      counter: a Counter which allows for recording of counts (learner steps,
        actor steps, etc.) distributed throughout the agent.
    """
        parent_counter = counter or counting.Counter()
        sub_learners = {}
        unzipped_dataset = iterator_utils.unzip_iterators(
            dataset, num_sub_iterators=self._num_agents
        )

        def make_logger_fn(agent_id: str) -> loggers.LoggerFactory:
            """Returns a logger factory for the subagent with the given id."""

            def logger_factory(
                label: loggers.LoggerLabel,
                steps_key: Optional[loggers.LoggerStepsKey] = None,
                instance: Optional[loggers.TaskInstance] = None,
            ) -> loggers.Logger:
                return logger_fn(f"{label}{agent_id}", steps_key, instance)

            return logger_factory

        for i_dataset, (agent_id, builder) in enumerate(self._builders.items()):
            counter = counting.Counter(parent_counter, prefix=f"{agent_id}")
            single_agent_spec = ma_utils.get_agent_spec(environment_spec, agent_id)
            random_key, learner_key = jax.random.split(random_key)
            sub_learners[agent_id] = builder.make_learner(
                learner_key,
                networks[agent_id],
                unzipped_dataset[i_dataset],
                logger_fn=make_logger_fn(agent_id),
                environment_spec=single_agent_spec,
                replay_client=replay_client,
                counter=counter,
            )
        return learner_set.SynchronousDecentralizedLearnerSet(
            sub_learners, separator=VARIABLE_SEPARATOR
        )

    def make_adder(  # Internal pytype check.
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        policy: Optional[ma_types.MultiAgentPolicyNetworks] = None,
    ) -> Mapping[ma_types.AgentID, Optional[adders.Adder]]:
        del environment_spec, policy  # Unused.
        return {
            agent_id: b.make_adder(replay_client, environment_spec=None, policy=None)
            for agent_id, b in self._builders.items()
        }

    def make_actor(  # Internal pytype check.
        self,
        random_key: networks_lib.PRNGKey,
        policy: ma_types.MultiAgentPolicyNetworks,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[Mapping[ma_types.AgentID, adders.Adder]] = None,
    ) -> core.Actor:
        """Returns simultaneous-acting multiagent actor instance.

    Args:
      random_key: random key.
      policy: dict of policies, one for each actor. Policies can
        be heterogeneous (i.e., distinct in architecture) across actors.
      environment_spec: the (multiagent) environment spec, which will be
        factorized into single-agent specs for replay table initialization.
      variable_source: an optional LearnerSet. Each sub_actor pulls its local
        variables from variable_source.
      adder: how data is recorded (e.g., added to replay) for each actor.
    """
        if adder is None:
            adder = {agent_id: None for agent_id in policy.keys()}

        sub_actors = {}
        for agent_id, builder in self._builders.items():
            single_agent_spec = ma_utils.get_agent_spec(environment_spec, agent_id)
            random_key, actor_key = jax.random.split(random_key)
            # Adds a prefix to each sub-actor's variable names to ensure the correct
            # sub-learner is queried for variables.
            sub_variable_source = PrefixedVariableSource(
                variable_source, f"{agent_id}{VARIABLE_SEPARATOR}"
            )
            sub_actors[agent_id] = builder.make_actor(
                actor_key,
                policy[agent_id],
                single_agent_spec,
                sub_variable_source,
                adder[agent_id],
            )
        return actor.SimultaneousActingMultiAgentActor(sub_actors)

    def make_policy(
        self,
        networks: ma_types.MultiAgentNetworks,
        environment_spec: specs.EnvironmentSpec,
        evaluation: bool = False,
    ) -> ma_types.MultiAgentPolicyNetworks:
        return decentralized_factories.policy_network_factory(
            networks,
            environment_spec,
            self._agent_types,
            self._agent_configs,
            eval_mode=evaluation,
            init_policy_network_fn=self._init_policy_network_fn,
        )
