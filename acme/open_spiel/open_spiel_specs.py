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

"""Objects which specify the extra information used by an OpenSpiel environment."""

from typing import Any, NamedTuple

from acme.open_spiel import open_spiel_wrapper


# TODO Move elsewhere? Not actually a spec.
class Extras(NamedTuple):
  """Extras used by a given environment."""
  legal_actions: Any
  terminals: Any


class ExtrasSpec(NamedTuple):
  """Full specification of the extras used by a given environment."""
  legal_actions: Any
  terminals: Any


def make_extras_spec(
    environment: open_spiel_wrapper.OpenSpielWrapper) -> ExtrasSpec:
  """Returns an `ExtrasSpec` describing additional values used by OpenSpiel."""
  return ExtrasSpec(legal_actions=environment.legal_actions_spec(),
                    terminals=environment.terminals_spec())
