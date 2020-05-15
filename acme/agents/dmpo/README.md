# Distributional Maximum a posteriori Policy Optimization (DMPO)

This folder contains an implementation of a novel agent (DMPO) introduced in
the original Acme release.
This work extends previous the MPO algorithm ([Abdolmaleki et al., 2018a],
[Abdolmaleki et al., 2018b]) by using a distributional Q-network similar to C51
([Bellemare et al., 2017]).

In this algorithm, the critic outputs a distribution over state-action values;
in this particular case this discrete distribution is parametrized as in C51.

[Abdolmaleki et al., 2018a]: https://arxiv.org/pdf/1806.06920.pdf
[Abdolmaleki et al., 2018b]: https://arxiv.org/pdf/1812.02256.pdf
[Bellemare et al., 2017]: https://arxiv.org/abs/1707.06887
