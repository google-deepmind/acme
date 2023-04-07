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

"""PPO agent."""

from acme.agents.jax.ppo.builder import PPOBuilder
from acme.agents.jax.ppo.config import PPOConfig
from acme.agents.jax.ppo.learning import PPOLearner
from acme.agents.jax.ppo.networks import (
    EntropyFn,
    PPONetworks,
    make_categorical_ppo_networks,
    make_continuous_networks,
    make_discrete_networks,
    make_inference_fn,
    make_mvn_diag_ppo_networks,
    make_networks,
    make_ppo_networks,
    make_tanh_normal_ppo_networks,
)
from acme.agents.jax.ppo.normalization import (
    NormalizationFns,
    NormalizedGenericActor,
    build_ema_mean_std_normalizer,
    build_mean_std_normalizer,
)
