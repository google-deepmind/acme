# Agents

Acme includes a number of pre-built agents listed below. We have also broken the
agents listed below into different sections based on their different use cases,
however these distinction are often subtle. For more information on each
implementation see the relevant agent-specific README.

**Continuous control.** Acme has long had a focus on continuous control agents
(i.e. settings where the action space consists of a continuous space). The
following agents focus on this setting:

-   [Deep Deterministic Policy Gradient (DDPG)](../acme/agents/ddpg/)
-   [Distributed Distributional Deep Determinist (D4PG)](../acme/agents/d4pg/)
-   [Maximum _a Posteriori_ Policy Optimisation (MPO)](../acme/agents/mpo/)
-   [Distributional Maximum _a Posteriori_ Policy Optimisation (DMPO)](../acme/agents/dmpo/)

**Discrete control.** We also include a number of agents built with discrete
action-spaces in mind. Note that the distinction between these agents and the
continuous agents listed can be somewhat arbitrary. E.g. Impala could be
implemented for continuous action spaces as well, but here we focus on a
discrete-action variant.

-   [Deep Q-Networks (DQN)](../acme/agents/dqn/)
-   [Recurrent Replay Distributed DQN (R2D2)](../acme/agents/r2d2/)
-   [Importance-Weighted Actor-Learner Architectures (IMPALA)](../acme/agents/impala/)

**Batch RL.** The structure of Acme also lends itself quite nicely to
"learner-only" algorithm for use in Batch RL (with no environment interactions).
Implemented algorithms include:

-   [Behavior Cloning (BC)](../acme/agents/bc/)

**From Demonstrations.** Acme also easily allows active data acquisition to be
combined with data from demonstrations. Such algorithms include:

-   [Deep Q-Networks from Demonstrations (DQfD)](../acme/agents/dqfd/)
-   [Recurrent Replay Distributed DQN from Demonstrations (R2D3)](../acme/agents/r2d3/)

**Model-based RL.** Finally, Acme also includes a variant of MCTS which can be
used for model-based RL using a given or learned simulator

-   [Monte Carlo Tree Search (MCTS)](../acme/agents/mcts/)
