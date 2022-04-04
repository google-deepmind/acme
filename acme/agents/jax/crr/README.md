# Critic Regularized Regression (CRR)

This folder contains an implementation of the CRR algorithm
([Wang et al., 2020]). It is an offline RL algorithm to learn policies from data
using a form of critic-regularized regression.

For the advantage estimate, a sampled mean is used. See policy.py file for
possible weighting coefficients for the policy loss (including exponential
estimated advantage). The policy network assumes a continuous action space.

[Wang et al., 2020]: https://arxiv.org/abs/2006.15134
