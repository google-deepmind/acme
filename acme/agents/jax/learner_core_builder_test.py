# python3
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

"""Tests for the builder interface."""
from typing import Any, Iterator, Optional

from absl.testing import absltest
from acme import core
from acme.agents.jax import learner_core_builder
from acme.jax import networks as networks_lib
from acme.utils import counting
import reverb


class GenericActorLearnerCoreBuilderTest(absltest.TestCase):

  def test_make_learner_override(self):
    with self.assertRaises(AssertionError):

      class MyBuilder(learner_core_builder.GenericActorLearnerCoreBuilder[Any,
                                                                          Any,
                                                                          Any,
                                                                          Any]):

        def make_learner(
            self,
            random_key: networks_lib.PRNGKey,
            networks: Any,
            dataset: Iterator[Any],
            replay_client: Optional[reverb.Client] = None,
            counter: Optional[counting.Counter] = None,
            checkpoint: bool = False,
        ) -> core.Learner:
          raise NotImplementedError()


if __name__ == '__main__':
  absltest.main()
