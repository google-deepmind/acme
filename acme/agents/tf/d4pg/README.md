# Distributed Distributional Deep Deterministic Policy Gradient (D4PG)

This folder contains an implementation of the D4PG agent introduced in
([Barth-Maron et al., 2018]), which extends previous Deterministic Policy
Gradient (DPG) algorithms ([Silver et al., 2014]; [Lillicrap et al., 2015]) by
using a distributional Q-network similar to C51 ([Bellemare et al., 2017]).

Note that since the synchronous agent is not distributed (i.e. not using
multiple asynchronous actors), it is not precisely speaking D4PG; a more
accurate name would be Distributional DDPG. In this algorithm, the critic
outputs a distribution over state-action values; in this particular case this
discrete distribution is parametrized as in C51.

Detailed notes:

-   The `vmin|vmax` hyperparameters of the distributional critic may need tuning
    depending on your environment's rewards. A good rule of thumb is to set
    `vmax` to the discounted sum of the maximum instantaneous rewards for the
    maximum episode length; then set `vmin` to `-vmax`.

[Barth-Maron et al., 2018]: https://arxiv.org/abs/1804.08617
[Bellemare et al., 2017]: https://arxiv.org/abs/1707.06887
[Lillicrap et al., 2015]: https://arxiv.org/abs/1509.02971
[Silver et al., 2014]: http://proceedings.mlr.press/v32/silver14
