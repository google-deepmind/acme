# Agents

Acme includes a number of pre-built agents listed below. We have also broken the
agents listed below into different sections based on their different use cases,
however these distinction are often subtle. For more information on each
implementation see the relevant agent-specific README.

**Continuous control.** Acme has long had a focus on continuous control agents
(i.e. settings where the action space consists of a continuous space). The
following agents focus on this setting:

Agent                                                          | Paper                    | Code
-------------------------------------------------------------- | :----------------------: | :--:
Deep Deterministic Policy Gradient (DDPG)                      | Lillicrap et al., 2015   | [![DDPG TF2](diagrams/tf.png){width="30" height="30"}][DDPG_TF2]
Distributed Distributional Deep Determinist (D4PG)             | Barth-Maron et al., 2018 | [![D4PG TF2](diagrams/tf.png){width="30" height="30"}][D4PG_TF2] [![D4PG JAX](diagrams/jax.png){width="52" height="30"}][D4PG_JAX]
Maximum a posteriori Policy Optimisation (MPO)                 | Abdolmaleki et al., 2018 | [![MPO TF2](diagrams/tf.png){width="30" height="30"}][MPO_TF2]
Distributional Maximum a posteriori Policy Optimisation (DMPO) | -                        | [![DMPO TF2](diagrams/tf.png){width="30" height="30"}][DMPO_TF2]

**Discrete control.** We also include a number of agents built with discrete
action-spaces in mind. Note that the distinction between these agents and the
continuous agents listed can be somewhat arbitrary. E.g. Impala could be
implemented for continuous action spaces as well, but here we focus on a
discrete-action variant.

Agent                                                    | Paper                    | Code
-------------------------------------------------------- | :----------------------: | :--:
Deep Q-Networks (DQN)                                    | Horgan et al., 2018      | [![DQN TF2](diagrams/tf.png){width="30" height="30"}][DQN_TF2] [![DQN JAX](diagrams/jax.png){width="52" height="30"}][DQN_JAX]
Importance-Weighted Actor-Learner Architectures (IMPALA) | Espeholt et al., 2018    | [![IMPALA TF2](diagrams/tf.png){width="30" height="30"}][IMPALA_TF2] [![IMPALA JAX](diagrams/jax.png){width="52" height="30"}][IMPALA_JAX]
Recurrent Replay Distributed DQN (R2D2)                  | Kapturowski et al., 2019 | [![R2D2 TF2](diagrams/tf.png){width="30" height="30"}][R2D2_TF2]

**Batch RL.** The structure of Acme also lends itself quite nicely to
"learner-only" algorithm for use in Batch RL (with no environment interactions).
Implemented algorithms include:

Agent | Paper | Code
----- | :---: | :----------------------------------------------------------:
BC    | -     | [![BC TF2](diagrams/tf.png){width="30" height="30"}][BC_TF2]

**From Demonstrations.** Acme also easily allows active data acquisition to be
combined with data from demonstrations. Such algorithms include:

| Agent | Paper    | Code                                                     |
| ----- | :------: | :------------------------------------------------------: |
| DQfD  | Hester   | [![DQFD                                                  |
:       : et al.,  : TF2](diagrams/tf.png){width="30" height="30"}][DQFD_TF2] :
:       : 2017     :                                                          :
| R2D3  | Gulcehre | [![R2D3                                                  |
:       : et al.,  : TF2](diagrams/tf.png){width="30" height="30"}][R2D3_TF2] :
:       : 2020     :                                                          :

**Model-based RL.** Finally, Acme also includes a variant of MCTS which can be
used for model-based RL using a given or learned simulator

Agent                          | Paper               | Code
------------------------------ | :-----------------: | :--:
Monte-Carlo Tree Search (MCTS) | Silver et al., 2018 | [![MCTS TF2](diagrams/tf.png){width="30" height="30"}][MCTS_TF2]

<!-- TF agents -->

[DQN_TF2]: ../acme/agents/dqn/
[IMPALA_TF2]: ../acme/agents/impala
[R2D2_TF2]: ../acme/agents/r2d2
[MCTS_TF2]: ../acme/agents/mcts
[DDPG_TF2]: ../acme/agents/ddpg
[D4PG_TF2]: ../acme/agents/d4pg
[MPO_TF2]: ../acme/agents/mpo
[DMPO_TF2]: ../acme/agents/dmpo
[BC_TF2]: ../acme/agents/bc
[DQFD_TF2]: ../acme/agents/dqfd
[R2D3_TF2]: ../acme/agents/r2d3

<!-- JAX agents -->

[DQN_JAX]: ../acme/agents/jax/dqn/
[IMPALA_JAX]: ../acme/agents/jax/impala/
[D4PG_JAX]: ../acme/agents/jax/d4pg/
