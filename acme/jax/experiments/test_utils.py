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

"""Utilities for testing of acme.jax.experiments functions."""

from acme.jax import experiments
from acme.tf import savers
from acme.utils import counting


def restore_counter(
    checkpointing_config: experiments.CheckpointingConfig,
) -> counting.Counter:
    """Restores a counter from the latest checkpoint saved with this config."""
    counter = counting.Counter()
    savers.Checkpointer(
        objects_to_save={"counter": counter},
        directory=checkpointing_config.directory,
        add_uid=checkpointing_config.add_uid,
        max_to_keep=checkpointing_config.max_to_keep,
    )
    return counter
