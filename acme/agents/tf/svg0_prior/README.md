# Stochastic Value Gradients (SVG) with Behavior Prior.

This folder contains a version of the SVG-0 agent introduced in
([Heess et al., 2015]) that has been extended with an entropy bonus, RETRACE
([Munos et al., 2016]) for off-policy correction and code to learn behavior
priors ([Tirumala et al., 2019], [Galashov et al., 2019]).

The base SVG-0 algorithm is similar to DPG and DDPG ([Silver et al., 2015],
[Lillicrap et al., 2015]) but uses the reparameterization trick to learn
stochastic and not deterministic policies. In addition, the RETRACE algorithm is
used to learn value functions using multiple timesteps of data with importance
sampling for off policy correction.

In addition an optional Behavior Prior can be learnt using this setup with an
information asymmetry that has shown to boost performance in some domains.
Example code to run with and without behavior priors on the DeepMind Control
Suite and Locomotion tasks are provided in the `examples` folder.

[Heess et al., 2015]: https://arxiv.org/abs/1510.09142
[Munos et al., 2016]: https://arxiv.org/abs/1606.02647
[Lillicrap et al., 2015]: https://arxiv.org/abs/1509.02971
[Silver et al., 2014]: http://proceedings.mlr.press/v32/silver14
[Tirumala et al., 2020]: https://arxiv.org/abs/2010.14274
[Galashov et al., 2019]: https://arxiv.org/abs/1905.01240
