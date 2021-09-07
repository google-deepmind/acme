# Deep Deterministic Policy Gradient (DDPG)

This folder contains an implementation of the DDPG agent introduced in (
[Lillicrap et al., 2015]), which extends the Deterministic Policy Gradient (DPG)
algorithm (introduced in [Silver et al., 2014]) to the realm of deep learning.

DDPG is an off-policy [actor-critic algorithm]. In this algorithm, critic is a
network that takes an observation and an action and outputs a value estimate
based on the current policy. It is trained to minimize the square
temporal-difference (TD) error. The actor is the policy network that takes
observations as input and outputs actions. For each observation, it is trained
to maximize the critic's value estimate.

[Lillicrap et al., 2015]: https://arxiv.org/abs/1509.02971
[Silver et al., 2014]: http://proceedings.mlr.press/v32/silver14
[actor-critic algorithm]: http://incompleteideas.net/book/RLbook2018.pdf#page=353
