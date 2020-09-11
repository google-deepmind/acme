# Agents

Acme includes a number of pre-built agents listed below. These are all
single-process agents. While there is currently no plan to release the
distributed variants of these agents, they share the exact same learning and
acting code as their single-process counterparts available in this repository.

We've also listed the agents below in separate sections based on their different
use cases, however these distinction are often subtle. For more information on
each implementation see the relevant agent-specific README.

### Continuous control

Acme has long had a focus on continuous control agents (i.e. settings where the
action space consists of a continuous space). The following agents focus on this
setting:

Agent                                                          | Paper                    | Code
-------------------------------------------------------------- | :----------------------: | :--:
Deep Deterministic Policy Gradient (DDPG)                      | Lillicrap et al., 2015   | [![TF](../../docs/logos/tf-small.png)][DDPG_TF2]
Distributed Distributional Deep Determinist (D4PG)             | Barth-Maron et al., 2018 | [![TF](../../docs/logos/tf-small.png)][D4PG_TF2]
Maximum a posteriori Policy Optimisation (MPO)                 | Abdolmaleki et al., 2018 | [![TF](../../docs/logos/tf-small.png)][MPO_TF2]
Distributional Maximum a posteriori Policy Optimisation (DMPO) | -                        | [![TF](../../docs/logos/tf-small.png)][DMPO_TF2]
<!-- Multi-Objective Maximum a posteriori Policy Optimisation (MO-MPO) | Abdolmaleki, Huang et al., 2020 | [![TF](../../docs/logos/tf-small.png)][MOMPO_TF2] -->

### Discrete control

We also include a number of agents built with discrete action-spaces in mind.
Note that the distinction between these agents and the continuous agents listed
can be somewhat arbitrary. E.g. Impala could be implemented for continuous
action spaces as well, but here we focus on a discrete-action variant.

Agent                                                    | Paper                    | Code
-------------------------------------------------------- | :----------------------: | :--:
Deep Q-Networks (DQN)                                    | Horgan et al., 2018      | [![TF](../../docs/logos/tf-small.png)][DQN_TF2] [![JAX](../../docs/logos/jax-small.png)][DQN_JAX]
Importance-Weighted Actor-Learner Architectures (IMPALA) | Espeholt et al., 2018    | [![TF](../../docs/logos/tf-small.png)][IMPALA_TF2] [![JAX](../../docs/logos/jax-small.png)][IMPALA_JAX]
Recurrent Replay Distributed DQN (R2D2)                  | Kapturowski et al., 2019 | [![TF](../../docs/logos/tf-small.png)][R2D2_TF2]

### Batch RL

The structure of Acme also lends itself quite nicely to "learner-only" algorithm
for use in Batch RL (with no environment interactions). Implemented algorithms
include:

Agent                 | Paper | Code
--------------------- | :---: | :---------------------------------:
Behavior Cloning (BC) | -     | [![TF](../../docs/logos/tf-small.png)][BC_TF2]

### Learning from demonstrations

Acme also easily allows active data acquisition to be combined with data from
demonstrations. Such algorithms include:

Agent                                                       | Paper                 | Code
----------------------------------------------------------- | :-------------------: | :--:
Deep Q-Learning from Demonstrations (DQfD)                  | Hester et al., 2017   | [![TF](../../docs/logos/tf-small.png)][DQFD_TF2]
Recurrent Replay Distributed DQN from Demonstratinos (R2D3) | Gulcehre et al., 2020 | [![TF](../../docs/logos/tf-small.png)][R2D3_TF2]

### Model-based RL

Finally, Acme also includes a variant of MCTS which can be used for model-based
RL using a given or learned simulator

Agent                          | Paper               | Code
------------------------------ | :-----------------: | :--:
Monte-Carlo Tree Search (MCTS) | Silver et al., 2018 | [![TF](../../docs/logos/tf-small.png)][MCTS_TF2]

<!-- TF agents -->

[DQN_TF2]: tf/dqn/
[IMPALA_TF2]: tf/impala/
[R2D2_TF2]: tf/r2d2/
[MCTS_TF2]: tf/mcts/
[DDPG_TF2]: tf/ddpg/
[D4PG_TF2]: tf/d4pg/
[MPO_TF2]: tf/mpo/
[DMPO_TF2]: tf/dmpo/
<!-- [MOMPO_TF2]: tf/mompo/ -->
[BC_TF2]: tf/bc/
[DQFD_TF2]: tf/dqfd/
[R2D3_TF2]: tf/r2d3/

<!-- JAX agents -->

[DQN_JAX]: jax/dqn/
[IMPALA_JAX]: jax/impala/
[D4PG_JAX]: jax/d4pg/
