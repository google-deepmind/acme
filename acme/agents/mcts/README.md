# Monte-Carlo Tree Search (MCTS)

This agent implements planning with a simulator (learned or otherwise), with
search guided by policy and value networks. This can be thought of as a
scaled-down and simplified version of the AlphaZero algorithm
([Silver et al., 2018]).

## Algorithm

We use a network-guided variant of UCT called pUCT.

The MCTS search policy is given by

$$a_{\mathrm{search}} = \arg\max_a\left[Q(s_t, a) + C P_\theta(s_t, a)\frac{\sqrt{N(s_t)}}{N(s_t, a) + 1}\right]$$

where:

-   $$C$$ is a constant hyperparameter.
-   $$P_\theta(s, a)$$ is a 'prior' policy network.
-   $$N(s)$$ and $$N(s, a)$$ are the state and state-action visit counts,
    respectively.

When planning, we bootstrap values from leaf notes in the search tree via a
learned value function $$V(s)$$ which is learned using standard TD-learning:

$$L_{\text{value}} = \Big[V(s_t) - \left(r_t + \gamma V(s_{t+1})\right)\Big]^2$$

The 'prior' policy network is learned via a distillation loss to immitate the
MCTS search policy:

$$L_{\text{policy}} = D_{\text{KL}}\Big(P(s, \cdot) \big\| MCTS(s)\Big),$$

where $$MCTS(s)$$ is the softmax distribution over the values at the root node
resulting from search.

The algorithm is agnostic to the choice of environment model -- this can be an
exact simulator (as in AlphaZero), or a learned transition model; we provide
examples of both cases.

[Silver et al., 2018]: https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
