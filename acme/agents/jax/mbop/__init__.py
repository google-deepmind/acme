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

from acme.agents.jax.mbop.acting import (
    ActorCore,
    make_actor,
    make_actor_core,
    make_ensemble_actor_core,
)
from acme.agents.jax.mbop.builder import MBOPBuilder
from acme.agents.jax.mbop.config import MBOPConfig
from acme.agents.jax.mbop.dataset import (
    EPISODE_RETURN,
    N_STEP_RETURN,
    episodes_to_timestep_batched_transitions,
    get_normalization_stats,
)
from acme.agents.jax.mbop.learning import (
    LoggerFn,
    MakeNStepReturnLearner,
    MakePolicyPriorLearner,
    MakeWorldModelLearner,
    MBOPLearner,
    TrainingState,
    make_ensemble_regressor_learner,
)
from acme.agents.jax.mbop.losses import (
    MBOPLosses,
    TransitionLoss,
    policy_prior_loss,
    world_model_loss,
)
from acme.agents.jax.mbop.models import (
    MakeNStepReturn,
    MakePolicyPrior,
    MakeWorldModel,
    make_ensemble_n_step_return,
    make_ensemble_policy_prior,
    make_ensemble_world_model,
)
from acme.agents.jax.mbop.mppi import (
    MPPIConfig,
    mppi_planner,
    return_top_k_average,
    return_weighted_average,
)
from acme.agents.jax.mbop.networks import (
    MBOPNetworks,
    make_networks,
    make_policy_prior_network,
    make_world_model_network,
)
