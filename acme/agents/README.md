# Agents

Acme includes a number of pre-built agents listed below. We have also broken the
agents listed below into different sections based on their different use cases,
however these distinction are often subtle. For more information on each
implementation see the relevant agent-specific README.

### Continuous control

Acme has long had a focus on continuous control agents (i.e. settings where the
action space consists of a continuous space). The following agents focus on this
setting:

Agent                                                          | Paper                    | Code
-------------------------------------------------------------- | :----------------------: | :--:
Deep Deterministic Policy Gradient (DDPG)                      | Lillicrap et al., 2015   | [![TF](../../docs/diagrams/tf.png)][DDPG_TF2]
Distributed Distributional Deep Determinist (D4PG)             | Barth-Maron et al., 2018 | [![TF](../../docs/diagrams/tf.png)][D4PG_TF2] [![JAX](../../docs/diagrams/jax.png)][D4PG_JAX]
Maximum a posteriori Policy Optimisation (MPO)                 | Abdolmaleki et al., 2018 | [![TF](../../docs/diagrams/tf.png)][MPO_TF2]
Distributional Maximum a posteriori Policy Optimisation (DMPO) | -                        | [![TF](../../docs/diagrams/tf.png)][DMPO_TF2]

### Discrete control

We also include a number of agents built with discrete action-spaces in mind.
Note that the distinction between these agents and the continuous agents listed
can be somewhat arbitrary. E.g. Impala could be implemented for continuous
action spaces as well, but here we focus on a discrete-action variant.

Agent                                                    | Paper                    | Code
-------------------------------------------------------- | :----------------------: | :--:
Deep Q-Networks (DQN)                                    | Horgan et al., 2018      | [![TF](../../docs/diagrams/tf.png)][DQN_TF2] [![JAX](../../docs/diagrams/jax.png)][DQN_JAX]
Importance-Weighted Actor-Learner Architectures (IMPALA) | Espeholt et al., 2018    | [![TF](../../docs/diagrams/tf.png)][IMPALA_TF2] [![JAX](../../docs/diagrams/jax.png)][IMPALA_JAX]
Recurrent Replay Distributed DQN (R2D2)                  | Kapturowski et al., 2019 | [![TF](../../docs/diagrams/tf.png)][R2D2_TF2]

### Batch RL

The structure of Acme also lends itself quite nicely to "learner-only" algorithm
for use in Batch RL (with no environment interactions). Implemented algorithms
include:

Agent | Paper | Code
----- | :---: | :------------------------------:
Behavior Cloning (BC)    | -     | [![TF](../../docs/diagrams/tf.png)][BC_TF2]

### Learning from demonstrations

Acme also easily allows active data acquisition to be combined with data from
demonstrations. Such algorithms include:

Agent | Paper                 | Code
----- | :-------------------: | :--------------------------------:
Deep Q-Learning from Demonstrations (DQfD)  | Hester et al., 2017   | [![TF](../../docs/diagrams/tf.png)][DQFD_TF2]
Recurrent Replay Distributed DQN from Demonstratinos (R2D3)  | Gulcehre et al., 2020 | [![TF](../../docs/diagrams/tf.png)][R2D3_TF2]

### Model-based RL

Finally, Acme also includes a variant of MCTS which can be used for model-based
RL using a given or learned simulator

Agent                          | Paper               | Code
------------------------------ | :-----------------: | :--:
Monte-Carlo Tree Search (MCTS) | Silver et al., 2018 | [![TF](../../docs/diagrams/tf.png)][MCTS_TF2]

<!-- TF agents -->

[DQN_TF2]: ../agents/dqn/
[IMPALA_TF2]: ../agents/impala
[R2D2_TF2]: ../agents/r2d2
[MCTS_TF2]: ../agents/mcts
[DDPG_TF2]: ../agents/ddpg
[D4PG_TF2]: ../agents/d4pg
[MPO_TF2]: ../agents/mpo
[DMPO_TF2]: ../agents/dmpo
[BC_TF2]: ../agents/bc
[DQFD_TF2]: ../agents/dqfd
[R2D3_TF2]: ../agents/r2d3

<!-- JAX agents -->

[DQN_JAX]: ../agents/jax/dqn/
[IMPALA_JAX]: ../agents/jax/impala/
[D4PG_JAX]: ../agents/jax/d4pg/
