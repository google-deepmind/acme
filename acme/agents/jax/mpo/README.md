# Maximum a posteriori Policy Optimization (MPO)

This folder contains an implementation of Maximum a posteriori Policy
Optimization (MPO) introduced in ([Abdolmaleki et al., 2018a], [2018b]). 
This implementation includes a few important options such as:

* distributional critics, including the categorical one introduced in the first
  version of the Acme paper—see `mpo.types.CriticType` for more details;
* categorical or Gaussian policies,
* mixed (or shared) experience replay ([Zhang & Sutton], [Schmitt et al.],
  [Hessel et al.]),
* per-dimension KL constraint satisfaction, and
* action penalization via the multi-objective MPO work of
  [Abdolmaleki, Huang et al., 2020].

Note: Unlike in the TF implementations of the MPO agent, here we do not separate
those with distributional critics.

Detailed notes:

* This agent performs efficient frame-stacking so the environment doesn't need
  to. Specifically, the actor and learner are both wrapped to stack frames so
  that the sequences of observations are unstacked and much lighter, both in
  transit and at rest in the replay buffer.
* This agent can be configured to use mixed replay by setting `replay_fraction`
  to a value in `(0, 1)` and setting `samples_per_insert = None`. This means the
  learner gets a fixed amount of the freshest on-policy experience as well as
  the replay experience.
* When using per-dimension KL constraint satisfaction, you may need to tune
  the value of `epsilon_mean` (and `epsilon_stddev` if not fixed). A good rule
  of thumb would be to divide it by the number of dimensions in the action
  space.

[Abdolmaleki et al., 2018a]: https://arxiv.org/pdf/1806.06920.pdf
[2018b]: https://arxiv.org/pdf/1812.02256.pdf
[Abdolmaleki, Huang et al., 2020]: https://arxiv.org/pdf/2005.07513.pdf
[Zhang & Sutton]: https://arxiv.org/abs/1712.01275
[Schmitt et al.]: https://arxiv.org/abs/1909.11583
[Hessel et al.]: https://arxiv.org/abs/2104.06159
