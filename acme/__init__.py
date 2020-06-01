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

"""Acme is a framework for reinforcement learning."""

# Expose specs and types modules.
from acme import specs
from acme import types

# Make __version__ accessible.
from acme._metadata import __version__

# Expose core interfaces.
from acme.core import Actor
# Internal core import.
from acme.core import Learner
from acme.core import Saveable
from acme.core import VariableSource

# Expose the environment loop.
from acme.environment_loop import EnvironmentLoop
# Internal environment_loop import.

from acme.specs import make_environment_spec

# Acme loves you.