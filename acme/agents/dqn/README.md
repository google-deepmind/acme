# Deep Q-Networks (DQN)

This folder contains an implementation of the DQN algorithm
([Mnih et al., 2013], [Mnih et al., 2015]), with extras bells & whistles:

-   N-step bootstrapping ([Sutton & Barto, 2018])
-   Double DQN ([van Hasselt et al., 2015])
-   Prioritized replay ([Schaul et al., 2015])

## Algorithm

We build the algorithm up in steps, similar to discussion in Rainbow DQN
([Hessel et al., 2017]).

-   **Q-learning with function approximation**. The loss $$\mathcal{L}$$ is
    given by the square temporal difference error:

    $$\mathcal{L} = \left[Q(s_t, a_t) - \left(r_t + \gamma \max_a Q'(s_{t+1}, a)\right)\right]^2$$

    For DQN, in addition:

    -   Note that $$Q'$$ is a _target network_ that is an older copy of the
        _online network_ $$Q$$, periodically copied.
    -   Transitions $$\left(s_t, a_t, r_t, s_{t+1}\right)$$ are sampled
        uniformly from a replay buffer.
    -   The action-value function $$Q$$ is parameterized by a feed-forward
        neural network.

-   **N-step bootstrapping**.

    $$\mathcal{L} = \left[Q(s_t, a_t) - \left(\sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n \max_aQ'(s_{t+n}, a)\right)\right]^2$$

    In addition, we substitute the Huber loss for the square loss in order to
    clip gradients over a certain value.

-   **Double Q-learning**. The loss becomes

    $$\mathcal{L} = \left[Q(s_t, a_t) - \left(\sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n Q'(s_{t+n}, \arg\max_aQ(s_{t+n}, a))\right)\right]^2$$

-   **Prioritized experience replay**. Each transition is sampled with
    probability proportional to the absolute TD error raised to some exponent
    $$\omega$$:

    $$p\propto \mathcal{L}^\frac{\omega}{2}$$

    In order to unbias the loss we scale by a corresponding importance sampling
    term.

## Implementation

## Experiments

### Atari

-   Network:

    -   The usual convnet + mlp.

-   Environment preprocessing:

    -   Frame downsampling (84, 84) and grayscaling
    -   Frame stacking (x4)
    -   Action repeat (x4)
    -   Pooling (for flicker)
    -   Zero discount on life loss

### BSuite

[Mnih et al., 2013]: https://arxiv.org/abs/1312.5602
[Mnih et al., 2015]: https://www.nature.com/articles/nature14236
[van Hasselt et al., 2015]: https://arxiv.org/abs/1509.06461
[Schaul et al., 2015]: https://arxiv.org/abs/1511.05952
[Hessel et al., 2017]: https://arxiv.org/abs/1710.02298
[Horgan et al., 2018]: https://arxiv.org/abs/1803.00933
[Sutton & Barto, 2018]: http://incompleteideas.net/book/the-book.html
