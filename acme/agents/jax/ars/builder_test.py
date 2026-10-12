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

"""Tests for the ARS builder."""

from acme import specs
from acme.agents.jax.ars import builder
from acme.agents.jax.ars import config as ars_config
from acme.agents.jax.ars import networks as ars_networks
from acme.testing import fakes

from absl.testing import absltest
from absl.testing import parameterized


class ARSBuilderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('behavior', False, ars_networks.BEHAVIOR_PARAMS_NAME),
      ('evaluation', True, ars_networks.EVAL_PARAMS_NAME),
  )
  def test_make_policy(self, evaluation, expected_params_name):
    environment = fakes.ContinuousEnvironment(
        action_dim=2, observation_dim=3, bounded=True)
    spec = specs.make_environment_spec(environment)
    networks = ars_networks.make_networks(spec)
    ars_builder = builder.ARSBuilder(ars_config.ARSConfig(), spec)

    params_name, policy_network = ars_builder.make_policy(
        networks, spec, evaluation=evaluation)

    self.assertEqual(params_name, expected_params_name)
    self.assertIs(policy_network, networks)


if __name__ == '__main__':
  absltest.main()
