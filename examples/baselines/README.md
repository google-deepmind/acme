# Acme baseline examples

This directory contains launcher scripts for the baselines referenced in [the paper](https://arxiv.org/abs/2006.00979).
These scripts reproduce the plots given in the paper.

## How to run

A command line for running the SAC baseline example in distributed mode on the Hopper environment limited to 100k environment steps:
```
cd examples/baselines/rl_continuous
python run_sac.py --run_distributed=True --env_name=gym:Hopper-v2 --num_steps=100_000
```
