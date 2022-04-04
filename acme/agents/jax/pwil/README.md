# PWIL

This folder contains an implementation of the PWIL algorithm
([R.Dadashi et al., 2020]).

The description of PWIL in ([R.Dadashi et al., 2020]) leaves the behavior
unspecified when the episode lengths are not fixed in advance. Here, we assign
zero reward when a trajectory exceeds the desired length, and keep the partial
return unaffected when a trajectory is shorter than the desired length.

We prefill the replay buffer in a concurrent thread of the learner, to avoid
potential Reverb deadlocks.

[R.Dadashi et al., 2020]: https://arxiv.org/abs/2006.04678
