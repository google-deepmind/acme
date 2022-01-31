# Twin Delayed Deep Deterministic policy gradient algorithm (TD3)

This folder contains an implementation of the TD3 algorithm,
[Fujimoto, 2018].


Note the following differences with the original author's implementation:

*   the default network architecture is a LayerNorm MLP,
*   there is no initial exploration phase with a random policy,
*   the target critic and twin critic updates are not delayed.

[Fujimoto, 2018]: https://arxiv.org/pdf/1802.09477.pdf
