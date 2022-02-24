# Deep Q-Networks (DQN)

This folder contains an implementation of the DQN algorithm
([Mnih et al., 2013], [Mnih et al., 2015]), with extras bells & whistles,
similar to Rainbow DQN ([Hessel et al., 2017]).

*   Q-learning with neural network function approximation. The loss is given by
    the Huber loss applied to the temporal difference error.
*   Target Q' network updated periodically ([Mnih et al., 2015]).
*   N-step bootstrapping ([Sutton & Barto, 2018]).
*   Double Q-learning ([van Hasselt et al., 2015]).
*   Prioritized experience replay ([Schaul et al., 2015]).

This DQN implementation has a configurable loss. In losses.py, you can find
ready-to-use implementations of other methods related to DQN.

* Vanilla Deep Q-learning [Mnih et al., 2013], with two optimization tweaks
  (Adam intead of RMSProp, square instead of Huber, as suggested e.g. by
  [Obando-Ceron et al., 2020]).
* Quantile regression DQN (QrDQN) [Dabney et al., 2017]
* Categorical DQN (C51) [Bellemare et al., 2017]
* Munchausen DQN [Vieillard et al., 2020]
* Regularized DQN (DQNReg) [Co-Reyes et al., 2021]


[Mnih et al., 2013]: https://arxiv.org/abs/1312.5602
[Mnih et al., 2015]: https://www.nature.com/articles/nature14236
[van Hasselt et al., 2015]: https://arxiv.org/abs/1509.06461
[Schaul et al., 2015]: https://arxiv.org/abs/1511.05952
[Bellemare et al., 2017]: https://arxiv.org/abs/1707.06887
[Dabney et al., 2017]: https://arxiv.org/abs/1710.10044
[Hessel et al., 2017]: https://arxiv.org/abs/1710.02298
[Horgan et al., 2018]: https://arxiv.org/abs/1803.00933
[Sutton & Barto, 2018]: http://incompleteideas.net/book/the-book.html
[Obando-Ceron et al., 2020]: https://arxiv.org/abs/2011.14826
[Vieillard et al., 2020]: https://arxiv.org/abs/2007.14430
[Co-Reyes et al., 2021]: https://arxiv.org/abs/2101.03958
