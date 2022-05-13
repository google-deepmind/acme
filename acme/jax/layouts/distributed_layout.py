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

"""Program definition for a distributed layout based on a builder."""

from typing import Callable, Dict, Optional, Sequence

from acme import core
from acme import specs
from acme.agents.jax import builders
from acme.jax import experiments
from acme.jax import inference_server
from acme.jax import types
from acme.utils import loggers
from acme.utils import observers as observers_lib
import launchpad as lp

# TODO(stanczyk): Remove when use cases are ported to the new location.
EvaluatorFactory = experiments.config.EvaluatorFactory
default_evaluator_factory = experiments.config.default_evaluator_factory
AgentNetwork = experiments.config.AgentNetwork
PolicyNetwork = experiments.config.PolicyNetwork
NetworkFactory = experiments.config.NetworkFactory
PolicyFactory = experiments.config.PolicyFactory
MakeActorFn = experiments.config.MakeActorFn
LoggerLabel = experiments.config.LoggerLabel
LoggerStepsKey = experiments.config.LoggerStepsKey
LoggerFn = experiments.config.LoggerFn
EvaluatorFactory = experiments.config.EvaluatorFactory

ActorId = int

SnapshotModelFactory = Callable[
    [experiments.config.AgentNetwork, specs.EnvironmentSpec],
    Dict[str, Callable[[core.VariableSource], types.ModelToSnapshot]]]

get_default_logger_fn = experiments.get_default_logger_fn
CheckpointingConfig = experiments.CheckpointingConfig


class DistributedLayout:
  """Program definition for a distributed agent based on a builder.

  DEPRECATED: Use make_distributed_experiment directly.
  """

  def __init__(
      self,
      seed: int,
      environment_factory: types.EnvironmentFactory,
      network_factory: experiments.config.NetworkFactory,
      builder: builders.GenericActorLearnerBuilder,
      policy_network: experiments.config.PolicyFactory,
      num_actors: int,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      actor_logger_fn: Optional[Callable[[ActorId], loggers.Logger]] = None,
      evaluator_factories: Sequence[experiments.config.EvaluatorFactory] = (),
      device_prefetch: bool = True,
      prefetch_size: int = 1,
      log_to_bigtable: bool = False,
      max_number_of_steps: Optional[int] = None,
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
      multithreading_colocate_learner_and_reverb: bool = False,
      checkpointing_config: Optional[CheckpointingConfig] = None,
      make_snapshot_models: Optional[SnapshotModelFactory] = None,
      inference_server_config: Optional[
          inference_server.InferenceServerConfig] = None):
    self._experiment_config = experiments.config.Config(
        builder=builder,
        environment_factory=environment_factory,
        environment_spec=environment_spec,
        network_factory=network_factory,
        policy_network_factory=policy_network,
        evaluator_factories=evaluator_factories,
        observers=observers,
        seed=seed,
        max_number_of_steps=max_number_of_steps,
        save_logs=log_to_bigtable)
    self._num_actors = num_actors
    self._actor_logger_fn = actor_logger_fn
    self._device_prefetch = device_prefetch
    self._prefetch_size = prefetch_size
    self._multithreading_colocate_learner_and_reverb = (
        multithreading_colocate_learner_and_reverb)
    self._checkpointing_config = checkpointing_config
    self._make_snapshot_models = make_snapshot_models
    self._inference_server_config = inference_server_config

  def build(self, name='agent', program: Optional[lp.Program] = None):
    """Build the distributed agent topology."""
    return experiments.make_distributed_experiment(
        self._experiment_config,
        self._num_actors,
        actor_logger_fn=self._actor_logger_fn,
        device_prefetch=self._device_prefetch,
        prefetch_size=self._prefetch_size,
        multithreading_colocate_learner_and_reverb=self
        ._multithreading_colocate_learner_and_reverb,
        checkpointing_config=self._checkpointing_config,
        make_snapshot_models=self._make_snapshot_models,
        inference_server_config=self._inference_server_config,
        name=name,
        program=program)
