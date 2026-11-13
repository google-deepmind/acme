# Copilot Instructions for Acme

## Build, Test, and Lint

- **Install**: `pip install .[jax,tf,testing,envs]`
- **Run Tests**: `pytest`
  - Single test: `pytest path/to/test.py`
  - Ignore agent tests (slow): `pytest --ignore-glob="*/*agent*_test.py"`
- **Type Check**: `pytype -k -j auto .`
- **Lint**: Follows Google Python Style Guide.

## High-Level Architecture

Acme is designed around the **Actor-Learner-Builder** pattern to support both single-process and distributed reinforcement learning.

- **Actor** (`acme.core.Actor`): Interacts with the environment. Methods: `select_action`, `observe`, `update`.
- **Learner** (`acme.core.Learner`): Consumes data (usually from replay) and updates model parameters. Methods: `step`.
- **EnvironmentLoop** (`acme.environment_loop.EnvironmentLoop`): Orchestrates the interaction between `Actor` and `dm_env.Environment`.
- **Builder** (`acme.agents.jax.builders.ActorLearnerBuilder`): Defines how to construct the `Actor`, `Learner`, `ReplayTables`, and `DatasetIterator` for a specific agent. This ensures components can be instantiated consistently in both local and distributed settings.
- **Replay**: Typically handled via `Reverb`. Agents define adders (e.g., `NStepTransitionAdder`) to insert data into replay tables.

## Key Conventions

- **Agent Structure**: Agents (e.g., in `acme/agents/jax/`) are typically organized into:
  - `config.py`: Dataclass for configuration (hyperparameters).
  - `builder.py`: Implementation of `ActorLearnerBuilder`.
  - `actor.py`: The actor implementation (often wraps a policy network).
  - `learning.py`: The learner implementation (loss functions, optimizer steps).
  - `networks.py`: Network definitions (often using Haiku or Flax for JAX).
- **Environment**: Uses `dm_env` interface (TimeStep with observation, reward, discount, step_type).
- **JAX vs TF**: Agents are strictly separated into `acme/agents/jax` and `acme/agents/tf`. New agents should prefer JAX.
- **Type Hinting**: Extensive use of type hints (`typing`, `acme.types`).
- **Logging**: Use `acme.utils.loggers`. `EnvironmentLoop` automatically logs metrics if provided a logger.
- **Adders**: Use `acme.adders` to handle data insertion into replay buffers.
