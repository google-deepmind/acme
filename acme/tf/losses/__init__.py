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

"""Various losses for training agent components (policies, critics, etc)."""

from acme.tf.losses.distributional import categorical
from acme.tf.losses.distributional import multiaxis_categorical
from acme.tf.losses.dpg import dpg
from acme.tf.losses.huber import huber
from acme.tf.losses.mompo import KLConstraint
from acme.tf.losses.mompo import MultiObjectiveMPO
from acme.tf.losses.mpo import MPO
from acme.tf.losses.r2d2 import transformed_n_step_loss

# Internal imports.
# pylint: disable=g-bad-import-order,g-import-not-at-top
from acme.tf.losses.quantile import NonUniformQuantileRegression
from acme.tf.losses.quantile import QuantileDistribution
