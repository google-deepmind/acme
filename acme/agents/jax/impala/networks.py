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

"""IMPALA networks definition."""

from acme import specs
from acme.jax import networks as networks_lib


IMPALANetworks = networks_lib.UnrollableNetwork


def make_atari_networks(env_spec: specs.EnvironmentSpec) -> IMPALANetworks:
  """Builds default IMPALA networks for Atari games."""

  def make_core_module() -> networks_lib.DeepIMPALAAtariNetwork:
    return networks_lib.DeepIMPALAAtariNetwork(env_spec.actions.num_values)

  return networks_lib.make_unrollable_network(env_spec, make_core_module)
