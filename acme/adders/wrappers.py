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

"""A library of useful adder wrappers."""

from typing import Iterable

from acme import types
from acme.adders import base
import dm_env


class ForkingAdder(base.Adder):
  """An adder that forks data into several other adders."""

  def __init__(self, adders: Iterable[base.Adder]):
    self._adders = adders

  def reset(self):
    for adder in self._adders:
      adder.reset()

  def add_first(self, timestep: dm_env.TimeStep):
    for adder in self._adders:
      adder.add_first(timestep)

  def add(self,
          action: types.NestedArray,
          next_timestep: dm_env.TimeStep,
          extras: types.NestedArray = ()):
    for adder in self._adders:
      adder.add(action, next_timestep, extras)


class IgnoreExtrasAdder(base.Adder):
  """An adder that ignores extras."""

  def __init__(self, adder: base.Adder):
    self._adder = adder

  def reset(self):
    self._adder.reset()

  def add_first(self, timestep: dm_env.TimeStep):
    self._adder.add_first(timestep)

  def add(self,
          action: types.NestedArray,
          next_timestep: dm_env.TimeStep,
          extras: types.NestedArray = ()):
    self._adder.add(action, next_timestep)
