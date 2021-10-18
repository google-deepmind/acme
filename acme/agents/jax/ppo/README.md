# Proximal Policy Optimization (PPO)

This folder contains an implementation of the PPO algorithm
([Schulman et al., 2017]) with clipped surrogate objective.

Implementation notes:

   - Reverb replay buffer is used so it is not strictly on-policy;
   - observations and rewards are not normalized.


[Schulman et al., 2017]: https://arxiv.org/abs/1707.06347
