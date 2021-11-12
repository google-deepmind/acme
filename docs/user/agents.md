# Agents

Acme includes a number of pre-built agents listed below. All are provided as
single-process agents, but we also include a distributed implementation using
[Launchpad](https://github.com/deepmind/launchpad). Distributed agents share
the exact same learning and acting code as their single-process counterparts
and can be executed either on a single machine
(--lp_launch_type=[local_mt|local_mp] command line flag for multi-threaded or
multi-process execution) or multi machine setup on GCP
(--lp_launch_type=vertex_ai). For details please refer to
[Launchpad documentation](https://github.com/deepmind/launchpad/search?q=%22class+LaunchType%22).

We've listed the agents below in separate sections based on their different
use cases, however these distinction are often subtle. For more information on
each implementation see the relevant agent-specific README.

## Continuous control

Acme has long had a focus on continuous control agents (i.e. settings where the
action space consists of a continuous space). The following agents focus on this
setting:

Agent                                                             | Paper                           | Code
----------------------------------------------------------------- | ------------------------------- | ----
Deep Deterministic Policy Gradient (DDPG)                         | Lillicrap et al., 2015          | [![TF]][DDPG_TF2]
Distributed Distributional Deep Determinist (D4PG)                | Barth-Maron et al., 2018        | [![TF]][D4PG_TF2]
Maximum a posteriori Policy Optimisation (MPO)                    | Abdolmaleki et al., 2018        | [![TF]][MPO_TF2]
Distributional Maximum a posteriori Policy Optimisation (DMPO)    | -                               | [![TF]][DMPO_TF2]
Multi-Objective Maximum a posteriori Policy Optimisation (MO-MPO) | Abdolmaleki, Huang et al., 2020 | [![TF]][MOMPO_TF2]

<br/>

## Discrete control

We also include a number of agents built with discrete action-spaces in mind.
Note that the distinction between these agents and the continuous agents listed
can be somewhat arbitrary. E.g. Impala could be implemented for continuous
action spaces as well, but here we focus on a discrete-action variant.

Agent                                                    | Paper                      | Code
-------------------------------------------------------- | -------------------------- | ----
Deep Q-Networks (DQN)                                    | [Horgan et al., 2018]      | [![TF]][DQN_TF2] [![JAX]][DQN_JAX]
Importance-Weighted Actor-Learner Architectures (IMPALA) | [Espeholt et al., 2018]    | [![TF]][IMPALA_TF2] [![JAX]][IMPALA_JAX]
Recurrent Replay Distributed DQN (R2D2)                  | [Kapturowski et al., 2019] | [![TF]][R2D2_TF2]

<br/>

## Batch RL

The structure of Acme also lends itself quite nicely to "learner-only" algorithm
for use in Batch RL (with no environment interactions). Implemented algorithms
include:

Agent                 | Paper | Code
--------------------- | ----- | --------------------------------
Behavior Cloning (BC) | -     | [![TF]][BC_TF2] [![JAX]][BC_JAX]

<br/>

## Learning from demonstrations

Acme also easily allows active data acquisition to be combined with data from
demonstrations. Such algorithms include:

Agent                                                       | Paper                 | Code
----------------------------------------------------------- | --------------------- | ----
Deep Q-Learning from Demonstrations (DQfD)                  | Hester et al., 2017   | [![TF]][DQFD_TF2]
Recurrent Replay Distributed DQN from Demonstratinos (R2D3) | Gulcehre et al., 2020 | [![TF]][R2D3_TF2]

<br/>

## Model-based RL

Finally, Acme also includes a variant of MCTS which can be used for model-based
RL using a given or learned simulator

Agent                          | Paper                 | Code
------------------------------ | --------------------- | -----------------
Monte-Carlo Tree Search (MCTS) | [Silver et al., 2018] | [![TF]][MCTS_TF2]

<br/>

<!-- Code logos -->

[TF]: logos/tf-small.png
[JAX]: logos/jax-small.png

<!-- TF agents -->

[DQN_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/dqn/
[IMPALA_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/impala/
[R2D2_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/r2d2/
[MCTS_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/mcts/
[DDPG_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/ddpg/
[D4PG_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/d4pg/
[MPO_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/mpo/
[DMPO_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/dmpo/
[MOMPO_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/mompo/
[BC_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/bc/
[DQFD_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/dqfd/
[R2D3_TF2]: https://github.com/deepmind/acme/blob/master/acme/agents/tf/r2d3/

<!-- JAX agents -->

[DQN_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/dqn/
[IMPALA_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/impala/
[D4PG_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/d4pg/
[BC_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/bc/

<!-- Papers -->

[Horgan et al., 2018]: https://arxiv.org/abs/1803.00933
[Silver et al., 2018]: https://science.sciencemag.org/content/362/6419/1140
[Espeholt et al., 2018]: https://arxiv.org/abs/1802.01561
[Kapturowski et al., 2019]: https://openreview.net/pdf?id=r1lyTjAqYX
