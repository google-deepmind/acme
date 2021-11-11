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

"""Utility function for building and launching launchpad programs."""

import functools
import inspect
import os
import sys
import time
from typing import Any, Callable

from absl import flags
from absl import logging
from acme.utils import counting
import launchpad as lp

FLAGS = flags.FLAGS


def partial_kwargs(function: Callable[..., Any],
                   **kwargs: Any) -> Callable[..., Any]:
  """Return a partial function application by overriding default keywords.

  This function is equivalent to `functools.partial(function, **kwargs)` but
  will raise a `ValueError` when called if either the given keyword arguments
  are not defined by `function` or if they do not have defaults.

  This is useful as a way to define a factory function with default parameters
  and then to override them in a safe way.

  Args:
    function: the base function before partial application.
    **kwargs: keyword argument overrides.

  Returns:
    A function.
  """
  # Try to get the argspec of our function which we'll use to get which keywords
  # have defaults.
  argspec = inspect.getfullargspec(function)

  # Figure out which keywords have defaults.
  if argspec.defaults is None:
    defaults = []
  else:
    defaults = argspec.args[-len(argspec.defaults):]

  # Find any keys not given as defaults by the function.
  unknown_kwargs = set(kwargs.keys()).difference(defaults)

  # Raise an error
  if unknown_kwargs:
    error_string = 'Cannot override unknown or non-default kwargs: {}'
    raise ValueError(error_string.format(', '.join(unknown_kwargs)))

  return functools.partial(function, **kwargs)


class StepsLimiter:
  """Process that terminates an experiment when `max_steps` is reached."""

  def __init__(self,
               counter: counting.Counter,
               max_steps: int,
               steps_key: str = 'actor_steps'):
    self._counter = counter
    self._max_steps = max_steps
    self._steps_key = steps_key

  def run(self):
    """Run steps limiter to terminate an experiment when max_steps is reached.
    """

    logging.info('StepsLimiter: Starting with max_steps = %d (%s)',
                 self._max_steps, self._steps_key)
    while True:
      # Update the counts.
      counts = self._counter.get_counts()
      num_steps = counts.get(self._steps_key, 0)

      logging.info('StepsLimiter: Reached %d recorded steps', num_steps)

      if num_steps > self._max_steps:
        logging.info('StepsLimiter: Max steps of %d was reached, terminating',
                     self._max_steps)
        lp.stop()

      # Don't spam the counter.
      for _ in range(10):
        # Do not sleep for a long period of time to avoid LaunchPad program
        # termination hangs (time.sleep is not interruptible).
        time.sleep(1)


# Resources for each individual instance of the program.
def make_xm_docker_resources(program: lp.Program):
  """Creates a dictionary containing Docker XManager resource requirements."""
  if (FLAGS.lp_launch_type != 'vertex_ai' and
      FLAGS.lp_launch_type != 'local_docker'):
    # Avoid importing 'xmanager' for local runs.
    return None

  # Reference lp.DockerConfig to force lazy import of xmanager by Launchpad and
  # then import it. It is done this way to avoid heavy imports by default.
  lp.DockerConfig  # pylint: disable=pointless-statement
  from xmanager import xm  # pylint: disable=g-import-not-at-top

  # Get number of each type of node.
  num_nodes = {k: len(v) for k, v in program.groups.items()}

  xm_resources = {}

  # Try to find location of GitHub-fetched Acme sources from which agent is
  # being launched. This is needed to determine packages to be installed inside
  # the Docker image and to copy the right version of Acme source code.

  acme_location = os.path.dirname(
      inspect.getframeinfo(sys._getframe(1)).filename)  # pylint: disable=protected-access
  while not os.path.exists(os.path.join(acme_location, 'setup.py')):
    acme_location = os.path.dirname(acme_location)
    if acme_location == '/':
      raise ValueError(
          'Failed to find Acme''s setup.py (needed to determine '
          'required packages to install inside Docker). Are you calling '
          'make_xm_docker_resources from within checked-out Acme repository?')

  tmp_dir = acme_location
  docker_requirements = os.path.join(acme_location, 'requirements.txt')

  # Extend PYTHONPATH with paths used by the launcher.
  python_path = []
  for path in sys.path:
    if path.startswith(acme_location) and acme_location != path:
      python_path.append(path[len(acme_location):])

  if 'replay' in num_nodes:
    replay_cpu = 6 + num_nodes.get('actor', 0) * 0.01
    replay_cpu = min(40, replay_cpu)

    xm_resources['replay'] = lp.DockerConfig(
        tmp_dir,
        docker_requirements,
        hw_requirements=xm.JobRequirements(cpu=replay_cpu, ram=10 * xm.GiB),
        python_path=python_path)

  if 'evaluator' in num_nodes:
    xm_resources['evaluator'] = lp.DockerConfig(
        tmp_dir,
        docker_requirements,
        hw_requirements=xm.JobRequirements(cpu=2, ram=4 * xm.GiB),
        python_path=python_path)

  if 'actor' in num_nodes:
    xm_resources['actor'] = lp.DockerConfig(
        tmp_dir,
        docker_requirements,
        hw_requirements=xm.JobRequirements(cpu=2, ram=4 * xm.GiB),
        python_path=python_path)

  if 'learner' in num_nodes:
    learner_cpu = 6 + num_nodes.get('actor', 0) * 0.01
    learner_cpu = min(40, learner_cpu)
    xm_resources['learner'] = lp.DockerConfig(
        tmp_dir,
        docker_requirements,
        hw_requirements=xm.JobRequirements(
            cpu=learner_cpu, ram=6 * xm.GiB, P100=1),
        python_path=python_path)

  if 'environment_loop' in num_nodes:
    xm_resources['environment_loop'] = lp.DockerConfig(
        tmp_dir,
        docker_requirements,
        hw_requirements=xm.JobRequirements(
            cpu=6, ram=6 * xm.GiB, P100=1),
        python_path=python_path)

  if 'counter' in num_nodes:
    xm_resources['counter'] = lp.DockerConfig(
        tmp_dir,
        docker_requirements,
        hw_requirements=xm.JobRequirements(cpu=3, ram=4 * xm.GiB),
        python_path=python_path)

  if 'cacher' in num_nodes:
    xm_resources['cacher'] = lp.DockerConfig(
        tmp_dir,
        docker_requirements,
        hw_requirements=xm.JobRequirements(cpu=3, ram=6 * xm.GiB),
        python_path=python_path)

  return xm_resources
