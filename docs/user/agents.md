# Agents

<!--* freshness: { owner: 'dm-acme-dev' reviewed: '2022-09-23' review_interval: '6 month' } *-->

Acme includes a number of pre-built agents listed below. All agents can be
used to run synchronous single-threaded or distributed experiments. Distributed
experiments are using [Launchpad](https://github.com/deepmind/launchpad)
and can be executed either on a single machine
(`--lp_launch_type=[local_mt|local_mp]` command line flag for multi-threaded or
multi-process execution) or multi machine setup on GCP
(`--lp_launch_type=vertex_ai`). For details please refer to
[Launchpad documentation](https://github.com/deepmind/launchpad/search?q=%22class+LaunchType%22).

We've listed the agents below in separate sections based on their different
use cases, however these distinction are often subtle. For more information on
each implementation see the relevant agent-specific README.

## Continuous control

Acme has long had a focus on continuous control agents (i.e. settings where the
action space is continuous). The following agents focus on this setting:

Agent                                                                 | Paper                             | Code
--------------------------------------------------------------------- | --------------------------------- | ----
Distributed Distributional Deep Deterministic Policy Gradients (D4PG) | [Barth-Maron et al., 2018]        | [![JAX]][D4PG_JAX] [![TF]][DDPG_TF2]
Twin Delayed Deep Deterministic policy gradient (TD3)                 | [Fujimoto, 2018.]                 | [![JAX]][TD3_JAX]
Soft Actor-Critic (SAC)                                               | [Haarnoja et al., 2018]           | [![JAX]][SAC_JAX]
Maximum a posteriori Policy Optimisation (MPO)                        | [Abdolmaleki et al., 2018]        | [![JAX]][MPO_JAX] [![TF]][MPO_TF2]
Proximal Policy Optimization (PPO)                                    | [Schulman et al., 2017]           | [![JAX]][PPO_JAX]
Distributional Maximum a posteriori Policy Optimisation (DMPO)        | -                                 | [![TF]][DMPO_TF2]
Multi-Objective Maximum a posteriori Policy Optimisation (MO-MPO)     | [Abdolmaleki, Huang et al., 2020] | [![TF]][MOMPO_TF2]

<br/>

## Discrete control

We also include a number of agents built with discrete action-spaces in mind.
Note that the distinction between these agents and the continuous agents listed
can be somewhat arbitrary. E.g. Impala could be implemented for continuous
action spaces as well, but here we focus on a discrete-action variant.

Agent                                                    | Paper                      | Code
-------------------------------------------------------- | -------------------------- | ----
Deep Q-Networks (DQN)                                    | [Horgan et al., 2018]      | [![JAX]][DQN_JAX] [![TF]][DQN_TF2]
Importance-Weighted Actor-Learner Architectures (IMPALA) | [Espeholt et al., 2018]    | [![JAX]][IMPALA_JAX] [![TF]][IMPALA_TF2] 
Recurrent Replay Distributed DQN (R2D2)                  | [Kapturowski et al., 2019] | [![JAX]][R2D2_JAX] [![TF]][R2D2_TF2]
Proximal Policy Optimization (PPO)                       | [Schulman et al., 2017]    | [![JAX]][PPO_JAX]

<br/>

## Offline RL

The structure of Acme also lends itself quite nicely to "learner-only" algorithm
for use in Offline RL (with no environment interactions). Implemented algorithms
include:

Agent                              | Paper                   | Code
---------------------------------- | ----------------------- | --------------------------------
Behavior Cloning (BC)              | [Pomerleau, 1991]       | [![JAX]][BC_JAX] [![TF]][BC_TF2]
Conservative Q-learning (CQL)      | [Kumar et al., 2020]    | [![JAX]][CQL_JAX]
Critic-Regularized Regressio (CRR) | [Wang et al., 2020]     | [![JAX]][CRR_JAX]
Behavior value estimation (BVE)    | [Gulcehre et al., 2021] | [![JAX]][BVE_JAX]

<br/>

## Imitation RL

Acme's modular interfaces simplify implementation of compositional agents, such
as imitation algorithms which include a direct RL method. Included are:

Agent                 | Paper                  | Code
--------------------- | ---------------------- | --------------------------------
AIL/DAC/GAIL          | [Ho and Ermon, 2016]   | [![JAX]][AIL_JAX]
SQIL                  | [Reddy et al., 2020]   | [![JAX]][SQIL_JAX]
PWIL                  | [Dadashi et al., 2021] | [![JAX]][PWIL_JAX]

<br/>

## Learning from demonstrations

In this setting, contrary to the Imitation RL, the environment has a
well-defined reward function and the demonstrations come with environment
rewards. 

Agent                                                                      | Paper                 | Code
-------------------------------------------------------------------------- | --------------------- | ----
Soft Actor-Critic from Demonstrations (SACfD)                              | -                     | [![JAX]][SACFD_JAX]
Twin Delayed Deep Deterministic policy gradient from Demonstrations (TD3fD)| -                     | [![JAX]][TD3FD_JAX]
Deep Q-Learning from Demonstrations (DQfD)                                 | [Hester et al., 2017] | [![TF]][DQFD_TF2]
Recurrent Replay Distributed DQN from Demonstrations (R2D3)                | [Paine et al., 2019]  | [![TF]][R2D3_TF2]

<br/>

## Model-based RL

Finally, Acme also includes a variant of MCTS which can be used for model-based
RL using a given or learned simulator

Agent                               | Paper                             | Code
----------------------------------- | --------------------------------- | -----------------
Model-Based Offline Planning (MBOP) | [Argenson and Dulac-Arnold, 2021] | [![JAX]][MBOP_JAX]
Monte-Carlo Tree Search (MCTS)      | [Silver et al., 2018]             | [![TF]][MCTS_TF2]

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

[AIL_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/ail/
[BC_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/bc/
[BVE_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/bve/
[CQL_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/cql/
[CRR_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/crr/
[D4PG_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/d4pg/
[DQN_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/dqn/
[IMPALA_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/impala/
[MBOP_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/mbop/
[MPO_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/mpo/
[PPO_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/ppo/
[PWIL_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/pwil/
[R2D2_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/r2d2/
[SAC_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/sac/
[SACFD_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/lfd/
[SQIL_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/sqil/
[TD3_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/td3/
[TD3FD_JAX]: https://github.com/deepmind/acme/blob/master/acme/agents/jax/lfd/

<!-- Papers -->

[Abdolmaleki et al., 2018]: https://arxiv.org/abs/1806.06920
[Abdolmaleki, Huang et al., 2020]: https://arxiv.org/abs/2005.07513
[Argenson and Dulac-Arnold, 2021]: https://arxiv.org/abs/2008.05556
[Barth-Maron et al., 2018]: https://arxiv.org/abs/1804.08617
[Dadashi et al., 2021]: https://arxiv.org/abs/2006.04678
[Espeholt et al., 2018]: https://arxiv.org/abs/1802.01561
[Fujimoto, 2018.]: https://arxiv.org/abs/1802.09477
[Fujimoto and Gu, 2021]: https://arxiv.org/abs/2106.06860
[Gulcehre et al., 2021]: https://arxiv.org/abs/2103.09575
[Hester et al., 2017]: https://arxiv.org/abs/1704.03732
[Haarnoja et al., 2018]: https://arxiv.org/abs/1801.01290
[Ho and Ermon, 2016]: https://arxiv.org/abs/1606.03476
[Horgan et al., 2018]: https://arxiv.org/abs/1803.00933
[Kapturowski et al., 2019]: https://openreview.net/pdf?id=r1lyTjAqYX
[Kumar et al., 2020]: https://arxiv.org/abs/2006.04779
[Paine et al., 2019]: `https://arxiv.org/abs/1909.01387
[Pomerleau, 1991]: https://www.ri.cmu.edu/pub_files/pub3/pomerleau_dean_1991_1/pomerleau_dean_1991_1.pdf
[Reddy et al., 2020]: https://arxiv.org/abs/1905.11108
[Schulman et al., 2017]: https://arxiv.org/abs/1707.06347
[Silver et al., 2018]: https://science.sciencemag.org/content/362/6419/1140
[Wang et al., 2020]: https://arxiv.org/abs/2006.15134
