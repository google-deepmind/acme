# Proximal Policy Optimization (PPO)

This folder contains an implementation of the PPO algorithm
([Schulman et al., 2017]) with clipped surrogate objective.

Implementation notes:
  - PPO is not a strictly on-policy algorithm. In each call to the learner's
    step function, a batch of transitions are taken from the Reverb replay
    buffer, and N epochs of updates are performed on the data in the batch.
    Using larger values for num_epochs and num_minibatches makes the algorithm
    "more off-policy".

[Schulman et al., 2017]: https://arxiv.org/abs/1707.06347
