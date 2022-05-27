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

"""Control suite environment factory."""

from typing import Optional

from acme import wrappers
import dm_env


def make_environment(
    evaluation: bool = False,
    domain_name: str = 'cartpole',
    task_name: str = 'balance',
    from_pixels: bool = False,
    frames_to_stack: int = 3,
    flatten_stack: bool = False,
    num_action_repeats: Optional[int] = None,
) -> dm_env.Environment:
  """Implements a control suite environment factory."""
  # Load dm_suite lazily not require Mujoco license when not using it.
  from dm_control import suite  # pylint: disable=g-import-not-at-top
  from acme.wrappers import mujoco as mujoco_wrappers  # pylint: disable=g-import-not-at-top

  # Load raw control suite environment.
  environment = suite.load(domain_name, task_name)

  # Maybe wrap to get pixel observations from environment state.
  if from_pixels:
    environment = mujoco_wrappers.MujocoPixelWrapper(environment)
    environment = wrappers.FrameStackingWrapper(
        environment, num_frames=frames_to_stack, flatten=flatten_stack)
  environment = wrappers.CanonicalSpecWrapper(environment, clip=True)

  if num_action_repeats:
    environment = wrappers.ActionRepeatWrapper(
        environment, num_repeats=num_action_repeats)
  environment = wrappers.SinglePrecisionWrapper(environment)

  if evaluation:
    # The evaluator in the distributed agent will set this to True so you can
    # use this clause to, e.g., set up video recording by the evaluator.
    pass

  return environment
