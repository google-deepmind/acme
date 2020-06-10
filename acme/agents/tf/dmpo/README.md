# Distributional Maximum a posteriori Policy Optimization (DMPO)

This folder contains an implementation of a novel agent (DMPO) introduced in
the original Acme release.
This work extends the MPO algorithm ([Abdolmaleki et al., 2018a], [2018b]) by
using a distributional Q-network similar to C51 ([Bellemare et al., 2017]).
Therefore, as in the case of the D4PG agent, this algorithm's critic outputs a
distribution over state-action values.

As in our MPO agent, this is a more general algorithm, the current implementation
targets the continuous control setting and is most readily applied to the
DeepMind control suite or similar control tasks. This implementation also
includes the options of:

- per-dimension KL constraint satisfaction, and
- action penalization via the multi-objective MPO work of
  [Abdolmaleki et al., 2020].

Detailed notes:

- The `vmin|vmax` hyperparameters of the distributional critic may need tuning
  depending on your environment's rewards. A good rule of thumb is to set `vmax`
  to the discounted sum of the maximum instantaneous rewards for the maximum
  episode length; then set `vmin` to `-vmax`.
- When using per-dimension KL constraint satisfaction, you may need to tune the
  value of `epsilon_mean` (and `epsilon_stddev` if not fixed). A good rule of
  thumb would be to divide it by the number of dimensions in the action space.

[Abdolmaleki et al., 2018a]: https://arxiv.org/pdf/1806.06920.pdf
[2018b]: https://arxiv.org/pdf/1812.02256.pdf
[Abdolmaleki et al., 2020]: https://arxiv.org/pdf/2005.07513.pdf
[Bellemare et al., 2017]: https://arxiv.org/abs/1707.06887
