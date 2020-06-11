<img src="docs/logos/acme.png" width="50%">

# Acme: A research framework for reinforcement learning

**[Overview](#overview)** | **[Installation](#installation)** |
**[Documentation]** | **[Agents]** | **[Examples]** | **[Paper]** |
**[Blog post]**

![PyPI Python Version](https://img.shields.io/pypi/pyversions/dm-acme)
![PyPI version](https://badge.fury.io/py/dm-acme.svg)
![pytest](https://github.com/deepmind/acme/workflows/pytest/badge.svg)

Acme is a library of reinforcement learning (RL) agents and agent building
blocks. Acme strives to expose simple, efficient, and readable agents, that
serve both as reference implementations of popular algorithms and as strong
baselines, while still providing enough flexibility to do novel research. The
design of Acme also attempts to provide multiple points of entry to the RL
problem at differing levels of complexity.

<div align="center" style="display: grid; grid-template-columns: auto auto;">
  <div>
    <video width="25%" autoplay loop muted>
      <source src="https://storage.googleapis.com/dm-acme/videos/d4pg_humanoid_run_features_short.webm" type="video/webm">
    </video>
    <video width="25%" autoplay loop muted>
      <source src="https://storage.googleapis.com/dm-acme/videos/d4pg_acrobot_swingup_features_short.webm" type="video/webm">
    </video>
  </div>
  <div>
    <video width="25%" autoplay loop muted>
      <source src="https://storage.googleapis.com/dm-acme/videos/r2d2_breakout.webm" type="video/webm">
    </video>
    <video width="25%" autoplay loop muted>
      <source src="https://storage.googleapis.com/dm-acme/videos/r2d2_ms_pacman.webm" type="video/webm">
    </video>
  </div>
</div>

## Overview

If you just want to get started using Acme quickly, the main thing to know about
the library is that we expose a number of agent implementations and an
`EnvironmentLoop` primitive that can be used as follows:

```python
loop = acme.EnvironmentLoop(environment, agent)
loop.run()
```

This will run a simple loop in which the given agent interacts with its
environment and learns from this interaction. This assumes an `agent` instance
(implementations of which you can find [here][Agents]) and an `environment`
instance which implements the [DeepMind Environment API][dm_env]. Each
individual agent also includes a `README.md` file describing the implementation
in more detail. Of course, these two lines of code definitely simplify the
picture. To actually get started, take a look at the detailed working code
examples found in our [examples] subdirectory which show how to instantiate a
few agents and environments. We also include a
[quickstart notebook][Quickstart].

Acme also tries to maintain this level of simplicity while either diving deeper
into the agent algorithms or by using them in more complicated settings. An
overview of Acme along with more detailed descriptions of its underlying
components can be found by referring to the [documentation]. And we also include
a [tutorial notebook][Tutorial] which describes in more detail the underlying
components behind a typical Acme agent and how these can be combined to form a
novel implementation.

> :information_source: Acme is first and foremost a framework for RL research written by
> researchers, for researchers. We use it for our own work on a daily basis. So
> with that in mind, while we will make every attempt to keep everything in good
> working order, things may break occasionally. But if so we will make our best
> effort to fix them as quickly as possible!

## Installation

We have tested `acme` on Python 3.6 & 3.7.

1.  **Optional**: We strongly recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies in order to avoid version conflicts:

    ```bash
    python3 -m venv acme
    source acme/bin/activate
    pip install --upgrade pip setuptools
    ```

1.  To install the core libraries (including [Reverb], our storage backend):

    ```bash
    pip install dm-acme
    pip install dm-acme[reverb]
    ```

1.  To install dependencies for our [JAX]- or [TensorFlow]-based agents:

    ```bash
    pip install dm-acme[tf]
    # and/or
    pip install dm-acme[jax]
    ```

1.  Finally, to install a few example environments (including [gym],
    [dm_control], and [bsuite]):

    ```bash
    pip install dm-acme[envs]
    ```

## Citing Acme

If you use Acme in your work, please cite the accompanying
[technical report][Paper]:

```bibtex
@article{hoffman2020acme,
    title={Acme: A Research Framework for Distributed Reinforcement Learning},
    author={Matt Hoffman and Bobak Shahriari and John Aslanides and Gabriel
        Barth-Maron and Feryal Behbahani and Tamara Norman and Abbas Abdolmaleki
        and Albin Cassirer and Fan Yang and Kate Baumli and Sarah Henderson and
        Alex Novikov and Sergio Gómez Colmenarejo and Serkan Cabi and Caglar
        Gulcehre and Tom Le Paine and Andrew Cowie and Ziyu Wang and Bilal Piot
        and Nando de Freitas},
    year={2020},
    journal={arXiv preprint arXiv:2006.00979},
    url={https://arxiv.org/abs/2006.00979},
}
```

[Agents]: acme/agents/
[Examples]: examples/
[Tutorial]: examples/tutorial.ipynb
[Quickstart]: examples/quickstart.ipynb
[Documentation]: docs/index.md
[Paper]: https://arxiv.org/abs/2006.00979
[Blog post]: https://deepmind.com/research/publications/Acme
[Reverb]: https://github.com/deepmind/reverb
[JAX]: https://github.com/google/jax
[TensorFlow]: https://tensorflow.org
[gym]: https://github.com/openai/gym
[dm_control]: https://github.com/deepmind/dm_env
[dm_env]: https://github.com/deepmind/dm_env
[bsuite]: https://github.com/deepmind/bsuite
