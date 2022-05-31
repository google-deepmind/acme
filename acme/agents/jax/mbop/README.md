# Model-Based Offline Planning (MBOP)

This folder contains an implementation of the MBOP algorithm ([Argenson and
Dulac-Arnold, 2021]). It is an offline RL algorithm that generates a model that
can be used to control the system directly through planning. The learning
components, i.e. the world model, policy prior and the n-step return, are simple
supervised ensemble learners. It uses the Model-Predictive Path Integral control
planner.

The networks assume continuous and flattened observation and action spaces. The
dataset, i.e. demonstrations, should be in timestep-batched format (i.e. triple
transitions of the previous, current and next timesteps) and normalized. See
dataset.py file for helper functions for loading RLDS datasets and
normalization.

[Argenson and Dulac-Arnold, 2021]: https://arxiv.org/abs/2008.05556
