# Adversarial Imitation Learning (AIL)

This folder contains a modular implementation of an Adversarial
Imitation Learning agent.
The initial algorithm is Generative Adversarial Imitation Learning
(GAIL - [Ho et al., 2016]), but many more tricks and variations are
available.
The corresponding paper ([Orsini et al., 2021]) explains and discusses
the utility of all those tricks.

AIL requires an off-policy RL algorithm to work, passed in as an
`ActorLearnerBuilder`.

If you use this code, please cite [Orsini et al., 2021].

[Ho et al., 2016]: https://arxiv.org/abs/1606.03476
[Orsini et al., 2021]: https://arxiv.org/abs/2106.00672
