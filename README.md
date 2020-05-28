# Acme

![pytest](https://github.com/deepmind/acme/workflows/pytest/badge.svg)

Acme is a library of reinforcement learning (RL) agents and agent building
blocks. Acme strives to expose simple, efficient, and readable agents, that
serve both as reference implementations of popular algorithms and as strong
baselines, while still providing enough flexibility to do novel research. The
design of Acme also attempts to provide multiple points of entry to the RL
problem at differing levels of complexity.

## Overview

At the highest level Acme exposes a number of agents which can be used simply as
follows:

```python
import acme

# Create an environment and an actor.
environment = ...
actor = ...

# Run the environment loop.
loop = acme.EnvironmentLoop(environment, actor)
loop.run()
```

however Acme also tries to maintain this level of simplicity while either diving
deeper into the agent algorithms or by using them in more complicated settings.
An overview of Acme along with more detailed descriptions of its underlying
components can be found by referring to the [documentation].

For a quick start, take a look at the more detailed working code examples found
in the [examples] subdirectory. And finally, for more information on the various
agent implementations available take a look at the [agents] subdirectory along
with the `README.md` associated with each agent.

[documentation]: docs/index.md
[examples]: examples/
[agents]: acme/agents/
