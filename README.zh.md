<img src="docs/imgs/acme.png" width="50%">

# Acme：面向强化学习的科研框架 (A research framework for reinforcement learning)

[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]
[![acme-tests][tests-badge]][tests]
[![Documentation Status][rtd-badge]][documentation]

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/dm-acme
[pypi-badge]: https://badge.fury.io/py/dm-acme.svg
[pypi]: https://pypi.org/project/dm-acme/
[tests-badge]: https://github.com/deepmind/acme/workflows/acme-tests/badge.svg
[tests]: https://github.com/deepmind/acme/actions/workflows/ci.yml
[rtd-badge]: https://readthedocs.org/projects/dm-acme/badge/?version=latest

Acme 是一个强化学习（RL）基础构建模块库，致力于提供简洁、高效且易于阅读的智能体（Agent）实现。首先，这些智能体既可作为标准参考实现，也能为算法性能评估提供强有力的基线（Baselines）。同时，Acme 所提供的基线智能体具备足够的灵活性与简洁性，可作为探索前沿创新研究的起点。最后，Acme 的各个构建模块经过精心设计，使智能体能够在不同规模下无缝运行（例如单流执行与分布式智能体）。

## 快速入门 (Getting started)

最快上手的方式是查看 [examples] 子目录中提供的详尽可运行代码示例。这些示例展示了如何实例化多种不同的智能体，并在各类环境中运行它们。若想更快速地体验单个智能体的使用，请参阅[快速入门笔记本][Quickstart]。有关智能体内部构造的更多细节，可参见我们的[教程笔记本][Tutorial]。此外，有关 Acme 及其底层组件的完整说明，请查阅[官方文档][documentation]。关于设计决策背后的更多背景信息与详细阐述，请参阅我们的[技术报告][Paper]。

> 注意：Acme 首先是由科研人员编写、面向科研人员的强化学习研究框架。我们自己在日常研究工作中每天都会使用它。因此，虽然我们会尽最大努力让所有功能保持良好的运行状态，但偶尔仍可能出现中断或变更；一旦发生此类情况，我们将尽全力以最快速度修复！

[examples]: examples/
[tutorial]: https://colab.research.google.com/github/deepmind/acme/blob/master/examples/tutorial.ipynb
[quickstart]: https://colab.research.google.com/github/deepmind/acme/blob/master/examples/quickstart.ipynb
[documentation]: https://dm-acme.readthedocs.io/
[paper]: https://arxiv.org/abs/2006.00979

## 安装 (Installation)

只需按照以下步骤操作即可快速完成安装并运行：

1.  虽然您可以在标准 Python 环境中安装 Acme，但我们**强烈**建议使用 [Python 虚拟环境](https://docs.python.org/3/tutorial/venv.html)来管理依赖项。这有助于避免版本冲突，并让整个安装过程更加顺畅。

    ```bash
    python3 -m venv acme
    source acme/bin/activate
    pip install --upgrade pip setuptools wheel
    ```

1.  虽然核心库 `dm-acme` 可以直接通过 pip 安装，但默认安装所包含的依赖集合是最小化的。具体而言，要运行内置的任何智能体，您还需要根据所选智能体安装 [JAX] 或 [TensorFlow]。因此，我们建议同时安装这些组件，即：

    ```bash
    pip install dm-acme[jax,tf]
    ```

1.  最后，若要安装一些示例环境（包括 [gym]、[dm_control] 和 [bsuite]）：

    ```bash
    pip install dm-acme[envs]
    ```

1.  **从 GitHub 安装**：如果您希望运行最前沿（bleeding-edge）版本的 Acme，可以克隆 Acme 的 GitHub 仓库，然后在主目录（即 `setup.py` 所在目录）下执行以下命令：

    ```bash
    pip install .[jax,tf,testing,envs]
    ```

## 引用 Acme (Citing Acme)

如果您在研究工作中使用了 Acme，请引用我们更新后的配套[技术报告][paper]：

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

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月19日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
