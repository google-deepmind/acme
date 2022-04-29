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

"""Tests for base."""

import copy
import pickle

from acme.testing import fakes
from acme.wrappers import base

from absl.testing import absltest


class BaseTest(absltest.TestCase):

  def test_pickle_unpickle(self):
    test_env = base.EnvironmentWrapper(environment=fakes.DiscreteEnvironment())

    test_env_pickled = pickle.dumps(test_env)
    test_env_restored = pickle.loads(test_env_pickled)
    self.assertEqual(
        test_env.observation_spec(),
        test_env_restored.observation_spec(),
    )

  def test_deepcopy(self):
    test_env = base.EnvironmentWrapper(environment=fakes.DiscreteEnvironment())
    copied_env = copy.deepcopy(test_env)
    del copied_env

if __name__ == '__main__':
  absltest.main()
