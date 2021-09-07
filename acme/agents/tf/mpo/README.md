# Maximum a posteriori Policy Optimization (MPO)

This folder contains an implementation of Maximum a posteriori Policy
Optimization (MPO) introduced in ([Abdolmaleki et al., 2018a], [2018b]). While
this is a more general algorithm, the current implementation targets the
continuous control setting and is most readily applied to the DeepMind control
suite or similar control tasks.

This implementation includes a few important options such as:

*   per-dimension KL constraint satisfaction, and
*   action penalization via the multi-objective MPO work of
    [Abdolmaleki, Huang et al., 2020].

See the DMPO agent directory for a similar agent that uses a distributional
critic. See the MO-MPO agent directory for an agent that optimizes for multiple
objectives.

Detailed notes:

*   When using per-dimension KL constraint satisfaction, you may need to tune
    the value of `epsilon_mean` (and `epsilon_stddev` if not fixed). A good rule
    of thumb would be to divide it by the number of dimensions in the action
    space.

[Abdolmaleki et al., 2018a]: https://arxiv.org/pdf/1806.06920.pdf
[2018b]: https://arxiv.org/pdf/1812.02256.pdf
[Abdolmaleki, Huang et al., 2020]: https://arxiv.org/pdf/2005.07513.pdf
