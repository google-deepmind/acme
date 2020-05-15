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

"""Common environment wrapper classes."""

from acme.wrappers.atari_wrapper import AtariWrapper
from acme.wrappers.base import wrap_all
from acme.wrappers.gym_wrapper import GymAtariAdapter
from acme.wrappers.gym_wrapper import GymWrapper
from acme.wrappers.observation_action_reward import ObservationActionRewardWrapper
from acme.wrappers.single_precision import SinglePrecisionWrapper
