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

"""Example running SAC on continuous control tasks."""

import helpers
import launchpad as lp
from absl import app, flags

from acme.agents.jax import td3
from acme.jax import experiments
from acme.utils import lp_utils

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "run_distributed",
    True,
    "Should an agent be executed in a distributed "
    "way. If False, will run single-threaded.",
)
flags.DEFINE_string("env_name", "gym:HalfCheetah-v2", "What environment to run")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_steps", 1_000_000, "Number of env steps to run.")
flags.DEFINE_integer("eval_every", 50_000, "How often to run evaluation.")
flags.DEFINE_integer("evaluation_episodes", 10, "Evaluation episodes.")


def build_experiment_config():
    """Builds TD3 experiment config which can be executed in different ways."""
    # Create an environment, grab the spec, and use it to create networks.

    suite, task = FLAGS.env_name.split(":", 1)
    network_factory = lambda spec: td3.make_networks(
        spec, hidden_layer_sizes=(256, 256, 256)
    )

    # Construct the agent.
    config = td3.TD3Config(policy_learning_rate=3e-4, critic_learning_rate=3e-4,)
    td3_builder = td3.TD3Builder(config)
    # pylint:disable=g-long-lambda
    return experiments.ExperimentConfig(
        builder=td3_builder,
        environment_factory=lambda seed: helpers.make_environment(suite, task),
        network_factory=network_factory,
        seed=FLAGS.seed,
        max_num_actor_steps=FLAGS.num_steps,
    )
    # pylint:enable=g-long-lambda


def main(_):
    config = build_experiment_config()
    if FLAGS.run_distributed:
        program = experiments.make_distributed_experiment(
            experiment=config, num_actors=4
        )
        lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
    else:
        experiments.run_experiment(
            experiment=config,
            eval_every=FLAGS.eval_every,
            num_eval_episodes=FLAGS.evaluation_episodes,
        )


if __name__ == "__main__":
    app.run(main)
