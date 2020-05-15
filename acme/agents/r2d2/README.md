# R2D2 - Recurrent Experience Replay in Distributed Reinforcement Learning

This folder contains an implementation of the R2D2 agent introduced in
([Kapturowski et al., 2019]). This work builds upon the DQN algorithm
([Mnih et al., 2013], [Mnih et al., 2015]) and Ape-X framework ([Horgan et al.,
2018]), extending distributed Q-Learning to use recurrent neural networks.

This version is a synchronous version of the agent, and is therefore not
distributed.

## Implementation details

The settings used are based on the R2D2 implementation in the original paper
([Kapturowski et al., 2019]). R2D2 reused some settings from the Ape-X paper
([Horgan et al. 2018]).

In other cases the settings have been modified to provide fast convergence,
which may be a concern in the synchronous where it is harder to run for the
large number of actor steps used in the original paper.

Specific settings:

-   **N-step bootstrapping** - N-step bootstrapping with $$n=5$$ is used to
    compute the target, matching the paper.

-   **LSTM settings** - Burn in length, trace length and replay period are set
    10, 32 and 5 respectively. These settings differ from the paper which used
    20, 80 and 40.

-   **Discount factor** - A discount factor of $$\gamma = 0.997$$ was used,
    matching the paper.

-   **Importance Sampling** - Importance sampling was used with a priority
    exponent of 0.9 and an importance sampling exponent of 0.6, matching the
    paper.

-   **Target Network Update** - The agent uses a Target Q network which is
    updated once every 2500 steps, matching the paper.

-   **Learning Rate** The learning rate is set to 1e-4, as used in the paper.

-   **Epsilon-greedy exploration** - The paper appears to default to the same
    epsilon-greedy exploration schedule as Ape-X, which means that each actor
    uses a fixed epsilon $$\epsilon = 0.4^k$$ where k is chosen uniformly from
    the range $$[1.0, 8.0]$$. This implementation uses a single synchronous
    agent, this has been implemented with a fixed epsilon of 0.05.

-   **Sample to insert ratio** - The ratio of transitions sampled to transitions
    inserted can affect performance. This value has been set to 0.25 as this
    value performed well in hyper-parameter sweeps.

-   **Evaluation** - An epsilon-greedy policy is used for evaluation, with a
    small fixed epsilon of 0.001 to prevent sub-optimal repetitions.

### Atari Preprocessing

Standard atari preprocessing settings were used:

-   Frame downsampling (84, 84) and grayscaling
-   Frame stacking (x4)
-   Action repeat (x4)
-   Pooling (for flicker)
-   Zero discount on life loss
-   Episode length is limited to 108,000 frames.

[Kapturowski et al., 2019]: https://openreview.net/forum?id=r1lyTjAqYX
[Mnih et al., 2013]: https://arxiv.org/abs/1312.5602
[Mnih et al., 2015]: https://www.nature.com/articles/nature14236
[Horgan et al. 2018]: https://arxiv.org/pdf/1803.00933
