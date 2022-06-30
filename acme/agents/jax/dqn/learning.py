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

from acme.adders import reverb as adders
from acme.agents.jax.dqn import learning_lib
from acme.agents.jax.dqn import losses
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import optax
import reverb


class DQNLearner(learning_lib.SGDLearner):
  """DQN learner.

  We are in the process of migrating towards a more general SGDLearner to allow
  for easy configuration of the loss. This is maintained now for compatibility.
  """

  def __init__(self,
               network: networks_lib.FeedForwardNetwork,
               discount: float,
               importance_sampling_exponent: float,
               target_update_period: int,
               iterator: Iterator[reverb.ReplaySample],
               optimizer: optax.GradientTransformation,
               random_key: networks_lib.PRNGKey,
               stochastic_network: bool = False,
               max_abs_reward: float = 1.,
               huber_loss_parameter: float = 1.,
               replay_client: Optional[reverb.Client] = None,
               replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None):
    """Initializes the learner."""
    loss_fn = losses.PrioritizedDoubleQLearning(
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        max_abs_reward=max_abs_reward,
        huber_loss_parameter=huber_loss_parameter,
        stochastic_network=stochastic_network,
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
    )
