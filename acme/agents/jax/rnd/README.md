# Random Network Distillation (RND)

This folder contains an implementation of the RND algorithm
([Burda et al., 2018])

RND requires a RL algorithm to work, passed in as an `ActorLearnerBuilder`.

By default this implementation ignores the original reward: the agent is trained
only on the intrinsic exploration reward. To also use extrinsic reward,
intrinsic and extrinsic reward weights can be passed into make_networks.

[Burda et al., 2018]: https://arxiv.org/abs/1810.12894
