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

"""Type aliases and assumptions that are specific to the MCTS agent."""

from typing import Callable, Tuple, Union
import numpy as np

# pylint: disable=invalid-name

# Assumption: actions are scalar and discrete (integral).
Action = Union[int, np.int32, np.int64]

# Assumption: observations are array-like.
Observation = np.ndarray

# Assumption: rewards and discounts are scalar.
Reward = Union[float, np.float32, np.float64]
Discount = Union[float, np.float32, np.float64]

# Notation: policy logits/probabilities are simply a vector of floats.
Probs = np.ndarray

# Notation: the value function is scalar-valued.
Value = float

# Notation: the 'evaluation function' maps observations -> (probs, value).
EvaluationFn = Callable[[Observation], Tuple[Probs, Value]]
