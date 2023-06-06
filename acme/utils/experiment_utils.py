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

"""Utility definitions for Acme experiments."""

from typing import Optional, Union

from acme.utils import loggers


def make_experiment_logger(label: str,
                           steps_key: Optional[str] = None,
                           task_instance: int = 0,
                           save_dir: Optional[Union[str, None]] = None,) -> loggers.Logger:
  # save_dir = "~/acme_experiment_utils"
  del task_instance
  if steps_key is None:
    steps_key = f'{label}_steps'
  return loggers.make_default_logger(label=label, steps_key=steps_key, save_dir=save_dir)


def create_experiment_logger_factory() -> loggers.LoggerFactory:
  return make_experiment_logger
