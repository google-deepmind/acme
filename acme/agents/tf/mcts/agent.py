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

"""A single-process MCTS agent."""

import numpy as np
import reverb
import sonnet as snt

from acme import datasets, specs
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf.mcts import acting, learning, models
from acme.tf import utils as tf2_utils


class MCTS(agent.Agent):
    """A single-process MCTS agent."""

    def __init__(
        self,
        network: snt.Module,
        model: models.Model,
        optimizer: snt.Optimizer,
        n_step: int,
        discount: float,
        replay_capacity: int,
        num_simulations: int,
        environment_spec: specs.EnvironmentSpec,
        batch_size: int,
    ):

        extra_spec = {
            "pi": specs.Array(
                shape=(environment_spec.actions.num_values,), dtype=np.float32
            )
        }
        # Create a replay server for storing transitions.
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=replay_capacity,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(
                environment_spec, extra_spec
            ),
        )
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f"localhost:{self._server.port}"
        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address), n_step=n_step, discount=discount
        )

        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(server_address=address)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        tf2_utils.create_variables(network, [environment_spec.observations])

        # Now create the agent components: actor & learner.
        actor = acting.MCTSActor(
            environment_spec=environment_spec,
            model=model,
            network=network,
            discount=discount,
            adder=adder,
            num_simulations=num_simulations,
        )

        learner = learning.AZLearner(
            network=network, optimizer=optimizer, dataset=dataset, discount=discount,
        )

        # The parent class combines these together into one 'agent'.
        super().__init__(
            actor=actor, learner=learner, min_observations=10, observations_per_step=1,
        )
