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

"""DQN learner implementation."""

from typing import Iterator, Optional

import optax
import reverb

from acme.adders import reverb as adders
from acme.agents.jax.dqn import learning_lib, losses
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting, loggers


class DQNLearner(learning_lib.SGDLearner):
    """DQN learner.

  We are in the process of migrating towards a more general SGDLearner to allow
  for easy configuration of the loss. This is maintained now for compatibility.
  """

    def __init__(
        self,
        network: networks_lib.TypedFeedForwardNetwork,
        discount: float,
        importance_sampling_exponent: float,
        target_update_period: int,
        iterator: Iterator[utils.PrefetchingSplit],
        optimizer: optax.GradientTransformation,
        random_key: networks_lib.PRNGKey,
        max_abs_reward: float = 1.0,
        huber_loss_parameter: float = 1.0,
        replay_client: Optional[reverb.Client] = None,
        replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        num_sgd_steps_per_step: int = 1,
    ):
        """Initializes the learner."""
        loss_fn = losses.PrioritizedDoubleQLearning(
            discount=discount,
            importance_sampling_exponent=importance_sampling_exponent,
            max_abs_reward=max_abs_reward,
            huber_loss_parameter=huber_loss_parameter,
        )
        super().__init__(
            network=network,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_iterator=iterator,
            target_update_period=target_update_period,
            random_key=random_key,
            replay_client=replay_client,
            replay_table_name=replay_table_name,
            counter=counter,
            logger=logger,
            num_sgd_steps_per_step=num_sgd_steps_per_step,
        )
