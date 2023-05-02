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

"""Shared helpers for different discrete RL experiment flavours."""

import functools
import subprocess
import os
import sys
from typing import Tuple

from absl import flags
from acme import specs
from acme import wrappers
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
from acme.jax import utils
import atari_py  # pylint:disable=unused-import
import dm_env
import gym
import haiku as hk
import jax.numpy as jnp


FLAGS = flags.FLAGS


def is_return_code_zero(args):
    """Return true iff the given command's return code is zero.
    All the messages to stdout or stderr are suppressed.
    """
    with open(os.devnull, "wb") as FNULL:
        try:
            subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        except subprocess.CalledProcessError:
            # The given command returned an error
            return False
        except OSError:
            # The given command was not found
            return False
        return True


def is_under_git_control():
    """Return true iff the current directory is under git control."""
    return is_return_code_zero(["git", "rev-parse"])


def save_git_information(outdir):
    # Save `git rev-parse HEAD` (SHA of the current commit)
    with open(os.path.join(outdir, "git-head.txt"), "wb") as f:
        f.write(subprocess.check_output("git rev-parse HEAD".split()))

    # Save `git status`
    with open(os.path.join(outdir, "git-status.txt"), "wb") as f:
        f.write(subprocess.check_output("git status".split()))

    # Save `git log`
    with open(os.path.join(outdir, "git-log.txt"), "wb") as f:
        f.write(subprocess.check_output("git log".split()))

    # Save `git diff`
    with open(os.path.join(outdir, "git-diff.txt"), "wb") as f:
        f.write(subprocess.check_output("git diff HEAD".split()))


def save_command_used(outdir):
  with open(os.path.join(outdir, "command.txt"), "w") as f:
      f.write(" ".join(sys.argv) + '\n')


def save_start_and_end_time(outdir, start_time, end_time):
  start_time_file = os.path.join(outdir, 'start_time.txt')
  end_time_file = os.path.join(outdir, 'end_time.txt')
  duration_file = os.path.join(outdir, 'duration_time.txt')
  with open(start_time_file, 'w') as f:
    f.write(str(start_time) + '\n')
  with open(end_time_file, 'w') as f:
    f.write(str(end_time) + '\n')
  with open(duration_file, 'w') as f:
    f.write(str(end_time - start_time) + '\n')


def make_atari_environment(
    level: str = 'Pong',
    sticky_actions: bool = True,
    zero_discount_on_life_loss: bool = False,
    oar_wrapper: bool = False,
    num_stacked_frames: int = 4,
    flatten_frame_stack: bool = False,
    grayscaling: bool = True,
    to_float: bool = True,
    scale_dims: Tuple[int, int] = (84, 84),
) -> dm_env.Environment:
  """Loads the Atari environment."""
# Internal logic.
  version = 'v0' if sticky_actions else 'v4'
  level_name = f'{level}NoFrameskip-{version}'
  env = gym.make(level_name, full_action_space=True)

  wrapper_list = [
      wrappers.GymAtariAdapter,
      functools.partial(
          wrappers.AtariWrapper,
          scale_dims=scale_dims,
          to_float=to_float,
          max_episode_len=108_000,
          num_stacked_frames=num_stacked_frames,
          flatten_frame_stack=flatten_frame_stack,
          grayscaling=grayscaling,
          zero_discount_on_life_loss=zero_discount_on_life_loss,
      ),
      wrappers.SinglePrecisionWrapper,
  ]

  if oar_wrapper:
    # E.g. IMPALA and R2D2 use this particular variant.
    wrapper_list.append(wrappers.ObservationActionRewardWrapper)

  return wrappers.wrap_all(env, wrapper_list)


def make_dqn_atari_network(
    environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
  """Creates networks for training DQN on Atari."""
  def network(inputs):
    model = hk.Sequential([
        networks_lib.AtariTorso(),
        hk.nets.MLP([512, environment_spec.actions.num_values]),
    ])
    return model(inputs)
  network_hk = hk.without_apply_rng(hk.transform(network))
  obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
  network = networks_lib.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, obs), apply=network_hk.apply)
  typed_network = networks_lib.non_stochastic_network_to_typed(network)
  return dqn.DQNNetworks(policy_network=typed_network)


def make_distributional_dqn_atari_network(
    environment_spec: specs.EnvironmentSpec,
    num_quantiles: int) -> dqn.DQNNetworks:
  """Creates networks for training Distributional DQN on Atari."""

  def network(inputs):
    model = hk.Sequential([
        networks_lib.AtariTorso(),
        hk.nets.MLP([512, environment_spec.actions.num_values * num_quantiles]),
    ])
    q_dist = model(inputs).reshape(-1, environment_spec.actions.num_values,
                                   num_quantiles)
    q_values = jnp.mean(q_dist, axis=-1)
    return q_values, q_dist

  network_hk = hk.without_apply_rng(hk.transform(network))
  obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
  network = networks_lib.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, obs), apply=network_hk.apply)
  typed_network = networks_lib.non_stochastic_network_to_typed(network)
  return dqn.DQNNetworks(policy_network=typed_network)
