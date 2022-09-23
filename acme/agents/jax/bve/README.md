# Behavior Value Estimation (BVE)

This folder contains the implementation BVE algorithm [1]. BVE is an offline RL 
algorithm that estimates the behavior value of the policy in the offline
dataset during the training. When deployed in an environment BVE does a single
step of policy improvement. It is a value based method. The original paper also
introduced regularizers to have conservative value estimates.

For simplicity of implementation the `rlax` sarsa loss function is used in 
`loss.py`. The network in `networks.py` is the typical DQN architecture.

[1] Gulcehre et al., Regularized Behavior Value Estimation, 2021, https://arxiv.org/abs/2103.09575.
