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

"""Program definition for a distributed layout for an offline RL experiment."""

from typing import Callable, Dict, Optional

from acme import core
from acme import specs
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import savers
from acme.jax import types
from acme.jax import utils
from acme.jax.experiments import config
from acme.jax import snapshotter
from acme.utils import counting
from acme.utils import lp_utils
import jax
import launchpad as lp


SnapshotModelFactory = Callable[[builders.Networks, specs.EnvironmentSpec],
                                Dict[str, Callable[[core.VariableSource],
                                                   types.ModelToSnapshot]]]


def make_distributed_offline_experiment(
    experiment: config.OfflineExperimentConfig,
    *,
    checkpointing_config: Optional[config.CheckpointingConfig] = None,
    make_snapshot_models: Optional[SnapshotModelFactory] = None,
    name='agent',
    program: Optional[lp.Program] = None):
  """Builds distributed agent based on a builder."""

  if checkpointing_config is None:
    checkpointing_config = config.CheckpointingConfig()

  def build_model_saver(variable_source: core.VariableSource):
    environment = experiment.environment_factory(0)
    spec = specs.make_environment_spec(environment)
    networks = experiment.network_factory(spec)
    models = make_snapshot_models(networks, spec)
    # TODO(raveman): Decouple checkpointing and snahpshotting configs.
    return snapshotter.JAXSnapshotter(
        variable_source=variable_source,
        models=models,
        path=checkpointing_config.directory,
        add_uid=checkpointing_config.add_uid)

  def build_counter():
    return savers.CheckpointingRunner(
        counting.Counter(),
        key='counter',
        subdirectory='counter',
        time_delta_minutes=5,
        directory=checkpointing_config.directory,
        add_uid=checkpointing_config.add_uid,
        max_to_keep=checkpointing_config.max_to_keep)

  def build_learner(
      random_key: networks_lib.PRNGKey,
      counter: Optional[counting.Counter] = None,
  ):
    """The Learning part of the agent."""

    dummy_seed = 1
    spec = (
        experiment.environment_spec or
        specs.make_environment_spec(experiment.environment_factory(dummy_seed)))

    # Creates the networks to optimize (online) and target networks.
    networks = experiment.network_factory(spec)

    dataset_key, random_key = jax.random.split(random_key)
    iterator = experiment.demonstration_dataset_factory(dataset_key)
    # make_demonstrations is responsible for putting data onto appropriate
    # training devices, so here we apply prefetch, so that data is copied over
    # in the background.
    iterator = utils.prefetch(iterable=iterator, buffer_size=1)
    counter = counting.Counter(counter, 'learner')
    learner = experiment.builder.make_learner(
        random_key=random_key,
        networks=networks,
        dataset=iterator,
        logger_fn=experiment.logger_factory,
        environment_spec=spec,
        counter=counter)

    learner = savers.CheckpointingRunner(
        learner,
        key='learner',
        subdirectory='learner',
        time_delta_minutes=5,
        directory=checkpointing_config.directory,
        add_uid=checkpointing_config.add_uid,
        max_to_keep=checkpointing_config.max_to_keep)

    return learner

  if not program:
    program = lp.Program(name=name)

  key = jax.random.PRNGKey(experiment.seed)

  counter = program.add_node(lp.CourierNode(build_counter), label='counter')

  if experiment.max_num_learner_steps is not None:
    program.add_node(
        lp.CourierNode(
            lp_utils.StepsLimiter,
            counter,
            experiment.max_num_learner_steps,
            steps_key='learner_steps'),
        label='counter')

  learner_key, key = jax.random.split(key)
  learner_node = lp.CourierNode(build_learner, learner_key, counter)
  learner = learner_node.create_handle()
  program.add_node(learner_node, label='learner')

  for evaluator in experiment.get_evaluator_factories():
    evaluator_key, key = jax.random.split(key)
    program.add_node(
        lp.CourierNode(evaluator, evaluator_key, learner, counter,
                       experiment.builder.make_actor),
        label='evaluator')

  if make_snapshot_models and checkpointing_config:
    program.add_node(lp.CourierNode(build_model_saver, learner),
                     label='model_saver')

  return program
