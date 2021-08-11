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

"""Implementations of a SVG0 agent with prior."""

from acme.agents.tf.svg0_prior.agent import SVG0
from acme.agents.tf.svg0_prior.agent_distributed import DistributedSVG0
from acme.agents.tf.svg0_prior.learning import SVG0Learner
from acme.agents.tf.svg0_prior.networks import make_default_networks
from acme.agents.tf.svg0_prior.networks import make_network_with_prior
