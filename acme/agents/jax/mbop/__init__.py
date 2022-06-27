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

"""Implementation of the Model-Based Offline Planning (MBOP) agent."""

from acme.agents.jax.mbop.acting import ActorCore
from acme.agents.jax.mbop.acting import make_actor
from acme.agents.jax.mbop.acting import make_actor_core
from acme.agents.jax.mbop.acting import make_ensemble_actor_core
from acme.agents.jax.mbop.dataset import EPISODE_RETURN
from acme.agents.jax.mbop.dataset import episodes_to_timestep_batched_transitions
from acme.agents.jax.mbop.dataset import get_normalization_stats
from acme.agents.jax.mbop.dataset import N_STEP_RETURN
from acme.agents.jax.mbop.learning import LoggerFn
from acme.agents.jax.mbop.learning import make_ensemble_regressor_learner
from acme.agents.jax.mbop.learning import MakeNStepReturnLearner
from acme.agents.jax.mbop.learning import MakePolicyPriorLearner
from acme.agents.jax.mbop.learning import MakeWorldModelLearner
from acme.agents.jax.mbop.learning import MBOPLearner
from acme.agents.jax.mbop.learning import TrainingState
from acme.agents.jax.mbop.losses import MBOPLosses
from acme.agents.jax.mbop.losses import policy_prior_loss
from acme.agents.jax.mbop.losses import TransitionLoss
from acme.agents.jax.mbop.losses import world_model_loss
from acme.agents.jax.mbop.models import make_ensemble_n_step_return
from acme.agents.jax.mbop.models import make_ensemble_policy_prior
from acme.agents.jax.mbop.models import make_ensemble_world_model
from acme.agents.jax.mbop.mppi import mppi_planner
from acme.agents.jax.mbop.mppi import MPPIConfig
from acme.agents.jax.mbop.mppi import return_top_k_average
from acme.agents.jax.mbop.mppi import return_weighted_average
from acme.agents.jax.mbop.networks import make_networks
from acme.agents.jax.mbop.networks import make_policy_prior_network
from acme.agents.jax.mbop.networks import make_world_model_network
from acme.agents.jax.mbop.networks import MBOPNetworks
