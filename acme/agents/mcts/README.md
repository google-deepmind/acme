# Monte-Carlo Tree Search (MCTS)

This agent implements planning with a simulator (learned or otherwise), with
search guided by policy and value networks. This can be thought of as a
scaled-down and simplified version of the AlphaZero algorithm
([Silver et al., 2018]).

The algorithm is agnostic to the choice of environment model -- this can be an
exact simulator (as in AlphaZero), or a learned transition model; we provide
examples of both cases.

[Silver et al., 2018]: https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
