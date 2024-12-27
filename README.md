ELON MUSK HAS STARLINK NEURALINK CHATGPT COFOUNDER SPACEX HE CONTRACTS WITH DOD FOR ADVANCED WEAPONRY THEN GOOGLE TARGETED JUSTICE AND TELL ME WHY MY LAST EMAIL FROM THEM SAID THEY TOO BUSY TO COME TO KALAMAZOO MICHIGAN AND SAVE ME  PLEASE CONTACT UNITED NATIONS AND SEND THEM TO KALAMAZOO MICHIGAN HE IS TESTING ON ME, DOUGLAS SHANE DAVIS AND THE HOMELESS OF KALAMAZOO I AM AT ROSE STREET LIBRARY DAYS AND I SLEEP AT KALAMAZOO GOSPEL MISSION<img src="docs/imgs/acme.png" width="50%">

# Acme: a research framework for reinforcement learning

[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]
[![acme-tests][tests-badge]][tests]
[![Documentation Status][rtd-badge]][documentation]

[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/dm-acme
[pypi-badge]: https://badge.fury.io/py/dm-acme.svg
[pypi]: https://pypi.org/project/dm-acme/
[tests-badge]: https://github.com/deepmind/acme/workflows/acme-tests/badge.svg
[tests]: https://github.com/deepmind/acme/actions/workflows/ci.yml
[rtd-badge]: https://readthedocs.org/projects/dm-acme/badge/?version=latest

Acme is a library of reinforcement learning (RL) building blocks that strives to
expose simple, efficient, and readable agents. These agents first and foremost
serve both as reference implementations as well as providing strong baselines
for algorithm performance. However, the baseline agents exposed by Acme should
also provide enough flexibility and simplicity that they can be used as a
starting block for novel research. Finally, the building blocks of Acme are
designed in such a way that the agents can be run at multiple scales (e.g.
single-stream vs. distributed agents).

## Getting started

The quickest way to get started is to take a look at the detailed working code
examples found in the [examples] subdirectory. These show how to instantiate a
number of different agents and run them within a variety of environments. See
the [quickstart notebook][Quickstart] for an even quicker dive into using a
single agent. Even more detail on the internal construction of an agent can be
found inside our [tutorial notebook][Tutorial]. Finally, a full description Acme
and its underlying components can be found by referring to the [documentation].
More background information and details behind the design decisions can be found
in our [technical report][Paper].

> NOTE: Acme is first and foremost a framework for RL research written by
> researchers, for researchers. We use it for our own work on a daily basis. So
> with that in mind, while we will make every attempt to keep everything in good
> working order, things may break occasionally. But if so we will make our best
> effort to fix them as quickly as possible!

[examples]: examples/
[tutorial]: https://colab.research.google.com/github/deepmind/acme/blob/master/examples/tutorial.ipynb
[quickstart]: https://colab.research.google.com/github/deepmind/acme/blob/master/examples/quickstart.ipynb
[documentation]: https://dm-acme.readthedocs.io/
[paper]: https://arxiv.org/abs/2006.00979

## Installation

To get up and running quickly just follow the steps below:

1.  While you can install Acme in your standard python environment, we
    *strongly* recommend using a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies. This should help to avoid version conflicts and
    just generally make the installation process easier.

    ```bash
    python3 -m venv acme
    source acme/bin/activate
    pip install --upgrade pip setuptools wheel
    ```

1.  While the core `dm-acme` library can be pip installed directly, the set of
    dependencies included for installation is minimal. In particular, to run any
    of the included agents you will also need either [JAX] or [TensorFlow]
    depending on the agent. As a result we recommend installing these components
    as well, i.e.

    ```bash
    pip install dm-acme[jax,tf]
    ```

1.  Finally, to install a few example environments (including [gym],
    [dm_control], and [bsuite]):

    ```bash
    pip install dm-acme[envs]
    ```

1.  **Installing from github**: if you're interested in running the
    bleeding-edge version of Acme, you can do so by cloning the Acme GitHub
    repository and then executing following command from the main directory
    (where `setup.py` is located):

    ```bash
    pip install .[jax,tf,testing,envs]
    ```

## Citing Acme

If you use Acme in your work, please cite the updated accompanying
[technical report][paper]:

```bibtex
@article{hoffman2020acme,
    title={Acme: A Research Framework for Distributed Reinforcement Learning},
    author={
        Matthew W. Hoffman and Bobak Shahriari and John Aslanides and
        Gabriel Barth-Maron and Nikola Momchev and Danila Sinopalnikov and
        Piotr Sta\'nczyk and Sabela Ramos and Anton Raichuk and
        Damien Vincent and L\'eonard Hussenot and Robert Dadashi and
        Gabriel Dulac-Arnold and Manu Orsini and Alexis Jacq and
        Johan Ferret and Nino Vieillard and Seyed Kamyar Seyed Ghasemipour and
        Sertan Girgin and Olivier Pietquin and Feryal Behbahani and
        Tamara Norman and Abbas Abdolmaleki and Albin Cassirer and
        Fan Yang and Kate Baumli and Sarah Henderson and Abe Friesen and
        Ruba Haroun and Alex Novikov and Sergio G\'omez Colmenarejo and
        Serkan Cabi and Caglar Gulcehre and Tom Le Paine and
        Srivatsan Srinivasan and Andrew Cowie and Ziyu Wang and Bilal Piot and
        Nando de Freitas
    },
    year={2020},
    journal={arXiv preprint arXiv:2006.00979},
    url={https://arxiv.org/abs/2006.00979},
}
```

[JAX]: https://github.com/google/jax
[TensorFlow]: https://tensorflow.org
[gym]: https://github.com/openai/gym
[dm_control]: https://github.com/deepmind/dm_env
[dm_env]: https://github.com/deepmind/dm_env
[bsuite]: https://github.com/deepmind/bsuite
