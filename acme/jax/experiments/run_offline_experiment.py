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

"""Runner used for executing local offline RL agents."""

import jax

import acme
from acme import specs
from acme.jax.experiments import config
from acme.tf import savers
from acme.utils import counting


def run_offline_experiment(
    experiment: config.OfflineExperimentConfig,
    eval_every: int = 100,
    num_eval_episodes: int = 1,
):
    """Runs a simple, single-threaded training loop using the default evaluators.

  It targets simplicity of the code and so only the basic features of the
  OfflineExperimentConfig are supported.

  Arguments:
    experiment: Definition and configuration of the agent to run.
    eval_every: After how many learner steps to perform evaluation.
    num_eval_episodes: How many evaluation episodes to execute at each
      evaluation step.
  """

    key = jax.random.PRNGKey(experiment.seed)

    # Create the environment and get its spec.
    environment = experiment.environment_factory(experiment.seed)
    environment_spec = experiment.environment_spec or specs.make_environment_spec(
        environment
    )

    # Create the networks and policy.
    networks = experiment.network_factory(environment_spec)

    # Parent counter allows to share step counts between train and eval loops and
    # the learner, so that it is possible to plot for example evaluator's return
    # value as a function of the number of training episodes.
    parent_counter = counting.Counter(time_delta=0.0)

    # Create the demonstrations dataset.
    dataset_key, key = jax.random.split(key)
    dataset = experiment.demonstration_dataset_factory(dataset_key)

    # Create the learner.
    learner_key, key = jax.random.split(key)
    learner = experiment.builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=dataset,
        logger_fn=experiment.logger_factory,
        environment_spec=environment_spec,
        counter=counting.Counter(parent_counter, prefix="learner", time_delta=0.0),
    )

    # Define the evaluation loop.
    eval_loop = None
    if num_eval_episodes > 0:
        # Create the evaluation actor and loop.
        eval_counter = counting.Counter(
            parent_counter, prefix="evaluator", time_delta=0.0
        )
        eval_logger = experiment.logger_factory(
            "evaluator", eval_counter.get_steps_key(), 0
        )
        eval_key, key = jax.random.split(key)
        eval_actor = experiment.builder.make_actor(
            random_key=eval_key,
            policy=experiment.builder.make_policy(networks, environment_spec, True),
            environment_spec=environment_spec,
            variable_source=learner,
        )
        eval_loop = acme.EnvironmentLoop(
            environment,
            eval_actor,
            counter=eval_counter,
            logger=eval_logger,
            observers=experiment.observers,
        )

    checkpointer = None
    if experiment.checkpointing is not None:
        checkpointing = experiment.checkpointing
        checkpointer = savers.Checkpointer(
            objects_to_save={"learner": learner, "counter": parent_counter},
            time_delta_minutes=checkpointing.time_delta_minutes,
            directory=checkpointing.directory,
            add_uid=checkpointing.add_uid,
            max_to_keep=checkpointing.max_to_keep,
            keep_checkpoint_every_n_hours=checkpointing.keep_checkpoint_every_n_hours,
            checkpoint_ttl_seconds=checkpointing.checkpoint_ttl_seconds,
        )

    max_num_learner_steps = experiment.max_num_learner_steps - parent_counter.get_counts().get(
        "learner_steps", 0
    )

    # Run the training loop.
    if eval_loop:
        eval_loop.run(num_eval_episodes)
    steps = 0
    while steps < max_num_learner_steps:
        learner_steps = min(eval_every, max_num_learner_steps - steps)
        for _ in range(learner_steps):
            learner.step()
            if checkpointer is not None:
                checkpointer.save()
        if eval_loop:
            eval_loop.run(num_eval_episodes)
        steps += learner_steps
