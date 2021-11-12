# Soft Actor-Critic (SAC)

This folder contains an implementation of the SAC algorithm
([Haarnoja et al., 2018]) with automatic tuning of the temperature
([Haarnoja et al., 2019]).

This is an actor-critic method with:

 - a stochastic policy optimization (as opposed to, e.g., DPG) with a maximum entropy regularization; and
 - two critics to mitigate the over-estimation bias in policy evaluation ([Fujimoto et al., 2018]).

For the maximum entropy regularization, we provide a commonly used heuristic for specifying entropy target (`target_entropy_from_env_spec`).
The heuristic returns `-num_actions` by default or `num_actions * target_entropy_per_dimension`
if `target_entropy_per_dimension` is specified.


[Haarnoja et al., 2018]: https://arxiv.org/abs/1801.01290
[Haarnoja et al., 2019]: https://arxiv.org/abs/1812.05905
[Fujimoto et al., 2018]: https://arxiv.org/abs/1802.09477
