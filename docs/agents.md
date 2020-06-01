# Agents

Acme includes a number of pre-built agents listed below. We have also broken the
agents listed below into different sections based on their different use cases,
however these distinction are often subtle. For more information on each
implementation see the relevant agent-specific README.

### Continuous control

Acme has long had a focus on continuous control agents (i.e. settings where the
action space consists of a continuous space). The following agents focus on this
setting:

| Agent             | Paper             | Code                                 |
| ----------------- | :---------------: | :----------------------------------: |
| Deep              | Lillicrap et al., | [![TF](diagrams/tf.png)][DDPG_TF2]   |
: Deterministic     : 2015              :                                      :
: Policy Gradient   :                   :                                      :
: (DDPG)            :                   :                                      :
| Distributed       | Barth-Maron et    | [![TF](diagrams/tf.png)][D4PG_TF2]   |
: Distributional    : al., 2018         : [![JAX](diagrams/jax.png)][D4PG_JAX] :
: Deep Determinist  :                   :                                      :
: (D4PG)            :                   :                                      :
| Maximum a         | Abdolmaleki et    | [![TF](diagrams/tf.png)][MPO_TF2]    |
: posteriori Policy : al., 2018         :                                      :
: Optimisation      :                   :                                      :
: (MPO)             :                   :                                      :
| Distributional    | -                 | [![TF](diagrams/tf.png)][DMPO_TF2]   |
: Maximum a         :                   :                                      :
: posteriori Policy :                   :                                      :
: Optimisation      :                   :                                      :
: (DMPO)            :                   :                                      :

### Discrete control

We also include a number of agents built with discrete action-spaces in mind.
Note that the distinction between these agents and the continuous agents listed
can be somewhat arbitrary. E.g. Impala could be implemented for continuous
action spaces as well, but here we focus on a discrete-action variant.

| Agent               | Paper        | Code                                   |
| ------------------- | :----------: | :------------------------------------: |
| Deep Q-Networks     | Horgan et    | [![TF](diagrams/tf.png)][DQN_TF2]      |
: (DQN)               : al., 2018    : [![JAX](diagrams/jax.png)][DQN_JAX]    :
| Importance-Weighted | Espeholt et  | [![TF](diagrams/tf.png)][IMPALA_TF2]   |
: Actor-Learner       : al., 2018    : [![JAX](diagrams/jax.png)][IMPALA_JAX] :
: Architectures       :              :                                        :
: (IMPALA)            :              :                                        :
| Recurrent Replay    | Kapturowski  | [![TF](diagrams/tf.png)][R2D2_TF2]     |
: Distributed DQN     : et al., 2019 :                                        :
: (R2D2)              :              :                                        :

### Batch RL

The structure of Acme also lends itself quite nicely to "learner-only" algorithm
for use in Batch RL (with no environment interactions). Implemented algorithms
include:

Agent | Paper | Code
----- | :---: | :------------------------------:
BC    | -     | [![TF](diagrams/tf.png)][BC_TF2]

### Learning from demonstrations

Acme also easily allows active data acquisition to be combined with data from
demonstrations. Such algorithms include:

Agent | Paper                 | Code
----- | :-------------------: | :--------------------------------:
DQfD  | Hester et al., 2017   | [![TF](diagrams/tf.png)][DQFD_TF2]
R2D3  | Gulcehre et al., 2020 | [![TF](diagrams/tf.png)][R2D3_TF2]

### Model-based RL

Finally, Acme also includes a variant of MCTS which can be used for model-based
RL using a given or learned simulator

| Agent            | Paper               | Code                               |
| ---------------- | :-----------------: | :--------------------------------: |
| Monte-Carlo Tree | Silver et al., 2018 | [![TF](diagrams/tf.png)][MCTS_TF2] |
: Search (MCTS)    :                     :                                    :

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
