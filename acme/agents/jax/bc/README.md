# Behavioral Cloning (BC)

This folder contains an implementation for supervised learning of a policy from
a dataset of observations and target actions.
This is an approach of Imitation Learning known as Behavioral Cloning,
introduced by [Pomerleau, 1989].

Several losses are implemented:

   * Mean squared error (mse)
   * Cross entropy (logp)
   * Peer Behavioral Cloning (peerbc), a regularization scheme from [Wang et al., 2021]
   * Reward-regularized Classification for Apprenticeship Learning (rcal), another
regularization scheme from [Piot et al., 2014], defined for discrete action
environments (or discretized action-spaces in case of continuous control).


[Pomerleau, 1989]: https://papers.nips.cc/paper/95-alvinn-an-autonomous-land-vehicle-in-a-neural-network.pdf
[Wang et al., 2021]: https://arxiv.org/pdf/2010.01748.pdf
[Piot et al., 2014]: https://www.cristal.univ-lille.fr/~pietquin/pdf/AAMAS_2014_BPMGOP.pdf
