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

"""Core Acme interfaces.

This file specifies and documents the notions of `Actor` and `Learner`.
"""

import abc
from typing import Generic, List, TypeVar

from acme import types
# Internal imports.
import dm_env

T = TypeVar('T')


class Actor(abc.ABC):
  """Interface for an agent that can act.

  This interface defines an API for an Actor to interact with an EnvironmentLoop
  (see acme.environment_loop), e.g. a simple RL loop where each step is of the
  form:

    # Make the first observation.
    timestep = env.reset()
    actor.observe_first(timestep.observation)

    # Take a step and observe.
    action = actor.select_action(timestep.observation)
    next_timestep = env.step(action)
    actor.observe(action, timestep)

    # Update the actor policy/parameters.
    actor.update()
  """

  @abc.abstractmethod
  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    """Samples from the policy and returns an action."""

  @abc.abstractmethod
  def observe_first(self, timestep: dm_env.TimeStep):
    """Make a first observation from the environment.

    Note that this need not be an initial state, it is merely beginning the
    recording of a trajectory.

    Args:
      timestep: first timestep.
    """

  @abc.abstractmethod
  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    """Make an observation of timestep data from the environment.

    Args:
      action: action taken in the environment.
      next_timestep: timestep produced by the environment given the action.
    """

  @abc.abstractmethod
  def update(self):
    """Perform an update of the actor parameters from past observations."""


# Internal class.


class VariableSource(abc.ABC):
  """Abstract source of variables.

  Objects which implement this interface provide a source of variables, returned
  as a collection of (nested) numpy arrays. Generally this will be used to
  provide variables to some learned policy/etc.
  """

  @abc.abstractmethod
  def get_variables(self, names: List[str]) -> List[types.NestedArray]:
    """Return the named variables as a collection of (nested) numpy arrays.

    Args:
      names: args where each name is a string identifying a predefined subset of
        the variables.

    Returns:
      A list of (nested) numpy arrays `variables` such that `variables[i]`
      corresponds to the collection named by `names[i]`.
    """


class Worker(abc.ABC):
  """An interface for (potentially) distributed workers."""

  @abc.abstractmethod
  def run(self):
    """Runs the worker."""


class Learner(VariableSource, Worker):
  """Abstract learner object.

  This corresponds to an object which implements a learning loop. A single step
  of learning should be implemented via the `update` method and this step
  is generally interacted with via the `run` method which runs update
  continuously.

  All objects implementing this interface should also be able to take in an
  external dataset (see acme.datasets) and run updates using data from this
  dataset. This can be accomplished by explicitly running `learner.update()`
  inside a for/while loop or by using the `learner.run()` convenience function.
  Data will be read from this dataset asynchronously and this is primarily
  useful when the dataset is filled by an external process.
  """

  @abc.abstractmethod
  def step(self):
    """Perform an update step of the learner's parameters."""

  def run(self):
    """Run the update loop; typically an infinite loop which calls step."""
    while True:
      self.step()


class Saveable(abc.ABC, Generic[T]):
  """An interface for saveable objects."""

  @abc.abstractmethod
  def save(self) -> T:
    """Returns the state from the object to be saved."""

  @abc.abstractmethod
  def restore(self, state: T):
    """Given the state, restores the object."""
