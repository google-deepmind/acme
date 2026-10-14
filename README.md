<img src="docs/imgs/acme.png" width="50%">


# Acme: A Research Framework for Reinforcement Learning

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dm-acme)
![PyPI](https://img.shields.io/pypi/v/dm-acme)
![Tests](https://img.shields.io/github/workflow/status/deepmind/acme/tests/main)
![Docs](https://img.shields.io/badge/docs-passing-brightgreen)

Acme is a flexible and scalable library for building reinforcement learning (RL) agents. It is designed with:

- 📚 **Clarity** – Easy-to-read code, ideal for learning and research
- 🧱 **Modularity** – Use individual building blocks or entire agents
- ⚙️ **Flexibility** – Supports both single-stream and distributed agents

Acme is used extensively in DeepMind’s research and aims to support both **reference implementations** and **novel algorithm development**.

---

## 🔍 Table of Contents
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Quick Example](#quick-example)
- [Examples](#examples)
- [Documentation & Tutorials](#documentation--tutorials)
- [Contributing](#contributing)
- [Citing Acme](#citing-acme)

---

## 🚀 Getting Started
The quickest way to get started is by exploring the working code examples in the `examples/` directory. These demonstrate how to instantiate and run various RL agents.

For an even quicker dive-in, try the [Quickstart Notebook](https://github.com/deepmind/acme/blob/master/docs/tutorial.ipynb).

---

## ⚙️ Installation
We recommend using a **Python virtual environment**:

```bash
python3 -m venv acme
source acme/bin/activate
pip install --upgrade pip setuptools wheel
```

Install Acme with recommended extras:

```bash
pip install dm-acme[jax,tf,envs]
```

To install from GitHub (for the latest version):

```bash
git clone https://github.com/deepmind/acme.git
cd acme
pip install .[jax,tf,testing,envs]
```

---

## 📅 Quick Example
Run a DQN agent in the `CartPole` environment:

```bash
python3 -m acme.examples.tf.dqn
```

For more examples, visit the `examples/` directory.

---

## 📖 Documentation & Tutorials
- 📄 [Official Documentation](https://github.com/deepmind/acme/tree/master/docs)
- 🧐 [Tutorial Notebook](https://github.com/deepmind/acme/blob/master/docs/tutorial.ipynb)
- 📊 [Technical Report](https://arxiv.org/abs/2006.00979)

---

## ✍️ Contributing
We welcome contributions from the community!

Start by checking for open issues, or try improving:
- Documentation
- Code examples
- New RL agents or algorithms

Please follow the standard GitHub flow: **fork > branch > commit > pull request**.

---

## 📄 Citing Acme
If you use Acme in your research, please cite:

```bibtex
@article{hoffman2020acme,
    title={Acme: A Research Framework for Distributed Reinforcement Learning},
    author={Matthew W. Hoffman and Bobak Shahriari and John Aslanides and Gabriel Barth-Maron and Nikola Momchev and Danila Sinopalnikov and Piotr Stanczyk and Sabela Ramos and Anton Raichuk and Damien Vincent and Leonard Hussenot and Robert Dadashi and Gabriel Dulac-Arnold and Manu Orsini and Alexis Jacq and Johan Ferret and Nino Vieillard and Seyed Kamyar Seyed Ghasemipour and Sertan Girgin and Olivier Pietquin and Feryal Behbahani and Tamara Norman and Abbas Abdolmaleki and Albin Cassirer and Fan Yang and Kate Baumli and Sarah Henderson and Abe Friesen and Ruba Haroun and Alex Novikov and Sergio Gomez Colmenarejo and Serkan Cabi and Caglar Gulcehre and Tom Le Paine and Srivatsan Srinivasan and Andrew Cowie and Ziyu Wang and Bilal Piot and Nando de Freitas},
    year={2020},
    journal={arXiv preprint arXiv:2006.00979},
    url={https://arxiv.org/abs/2006.00979},
}
```

---

Happy Reinforcement Learning! 🌟

