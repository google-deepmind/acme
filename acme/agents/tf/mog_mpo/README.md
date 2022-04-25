# Mixture of Gaussian distributional MPO (MoG-DMPO)

This folder contains an implementation of a novel agent (MoG-MPO) introduced in
[this technical report](https://arxiv.org/abs/2204.10256). 
This work extends the MPO algorithm ([Abdolmaleki et al., 2018a], [2018b]) by
using a distributional Q-network parameterized as a mixture of Gaussians.
Therefore, as in the case of the D4PG and DMPO agent, this algorithm's critic
outputs a distribution over state-action values.

As in our MPO agent, this is a more general algorithm, the current
implementation targets the continuous control setting and is most readily
applied to the DeepMind control suite or similar control tasks. This
implementation also includes the options of:

*   per-dimension KL constraint satisfaction, and
*   action penalization via the multi-objective MPO work of
    [Abdolmaleki et al., 2020].

Detailed notes:

*   When using per-dimension KL constraint satisfaction, you may need to tune
    the value of `epsilon_mean` (and `epsilon_stddev` if not fixed). A good rule
    of thumb would be to divide it by the number of dimensions in the action
    space.

[Abdolmaleki et al., 2018a]: https://arxiv.org/pdf/1806.06920.pdf
[2018b]: https://arxiv.org/pdf/1812.02256.pdf
[Abdolmaleki et al., 2020]: https://arxiv.org/pdf/2005.07513.pdf

Citation:

```
@misc{mog_mpo,
  title = {Revisiting Gaussian mixture critics in off-policy reinforcement
           learning: a sample-based approach},
  url = {https://arxiv.org/abs/2204.10256},
  author = {Shahriari, Bobak and
            Abdolmaleki, Abbas and
            Byravan, Arunkumar and
            Friesen, Abram and
            Liu, Siqi and
            Springenberg, Jost Tobias and
            Heess, Nicolas and
            Hoffman, Matt and
            Riedmiller, Martin},
  publisher = {arXiv},
  year = {2022},
}
```
