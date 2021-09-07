# Multi-Objective Maximum a posteriori Policy Optimization (MO-MPO)

This folder contains an implementation of Multi-Objective Maximum a posteriori
Policy Optimization (MO-MPO), introduced in ([Abdolmaleki, Huang et al., 2020]).
This trains a policy that optimizes for multiple objectives, with the desired
preference across objectives encoded by the hyperparameters `epsilon`.

As with our MPO agent, while this is a more general algorithm, the current
implementation targets the continuous control setting and is most readily
applied to the DeepMind control suite or similar control tasks. This
implementation also includes the options of:

*   per-dimension KL constraint satisfaction, and
*   distributional (per-objective) critics, as used by the DMPO agent

Detailed notes:

*   When using per-dimension KL constraint satisfaction, you may need to tune
    the value of `epsilon_mean` (and `epsilon_stddev` if not fixed). A good rule
    of thumb would be to divide it by the number of dimensions in the action
    space.
*   If using a distributional critic, the `vmin|vmax` hyperparameters of the
    distributional critic may need tuning depending on your environment's
    rewards. A good rule of thumb is to set `vmax` to the discounted sum of the
    maximum instantaneous rewards for the maximum episode length; then set
    `vmin` to `-vmax`.

[Abdolmaleki, Huang et al., 2020]: https://arxiv.org/abs/2005.07513
