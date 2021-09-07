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

[Mnih et al., 2013]: https://arxiv.org/abs/1312.5602
[Mnih et al., 2015]: https://www.nature.com/articles/nature14236
[van Hasselt et al., 2015]: https://arxiv.org/abs/1509.06461
[Schaul et al., 2015]: https://arxiv.org/abs/1511.05952
[Hessel et al., 2017]: https://arxiv.org/abs/1710.02298
[Horgan et al., 2018]: https://arxiv.org/abs/1803.00933
[Sutton & Barto, 2018]: http://incompleteideas.net/book/the-book.html
