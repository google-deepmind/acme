# Implicit Q-Learning (IQL)

This directory contains an implementation of Implicit Q-Learning (IQL), an
offline reinforcement learning algorithm.

## Overview

IQL is designed for learning from fixed datasets without online interaction.
Unlike other offline RL methods, IQL avoids querying values of out-of-sample
actions, which helps prevent overestimation and distributional shift issues.

## Key Features

- **Expectile Regression**: Uses expectile regression to learn a value function
  that approximates the upper expectile of Q-values, implicitly estimating the
  value of the best actions.

- **No Out-of-Sample Queries**: Never evaluates actions outside the dataset,
  avoiding distributional shift problems.

- **Advantage-Weighted Regression**: Extracts the policy using advantage-
  weighted behavioral cloning, which maximizes Q-values while staying close to
  the data distribution.

## Algorithm Components

1. **Value Function (V)**: Trained with expectile regression to estimate state
   values as an upper expectile of Q-values.

2. **Q-Function**: Trained with standard TD learning using the value function
   for next state values.

3. **Policy**: Trained with advantage-weighted regression to maximize Q-values
   weighted by advantages.

## Usage

```python
from acme.agents.jax import iql
from acme import specs

# Create networks
environment_spec = specs.make_environment_spec(environment)
networks = iql.make_networks(environment_spec)

# Configure IQL
config = iql.IQLConfig(
    expectile=0.7,  # Higher values are more conservative
    temperature=3.0,  # Higher values give more weight to high-advantage actions
    batch_size=256,
)

# Create builder
builder = iql.IQLBuilder(config)

# Create learner
learner = builder.make_learner(
    random_key=jax.random.PRNGKey(0),
    networks=networks,
    dataset=dataset_iterator,
    logger_fn=logger_factory,
    environment_spec=environment_spec,
)
```

## Hyperparameters

- **expectile** (default: 0.7): Controls the expectile for value function.
  Values > 0.5 give upper expectiles. Higher values (e.g., 0.9) are more
  conservative.

- **temperature** (default: 3.0): Inverse temperature for advantage weighting.
  Higher values give more weight to high-advantage actions.

- **tau** (default: 0.005): Target network update coefficient (Polyak
  averaging).

- **discount** (default: 0.99): Discount factor for TD updates.

## References

Kostrikov, I., Nair, A., & Levine, S. (2021). Offline Reinforcement Learning
with Implicit Q-Learning. arXiv preprint arXiv:2110.06169.

https://arxiv.org/abs/2110.06169
