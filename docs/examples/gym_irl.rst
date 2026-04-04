Gym Environment IRL Benchmarks
==============================

These examples run neural IRL estimators on standard OpenAI Gym control environments where the state space is continuous and tabular methods cannot be applied. The benchmark script runs a head-to-head comparison of NeuralGLADIUS and NeuralAIRL on CartPole, Acrobot, and LunarLander under low-data and high-data conditions following the experimental protocol from Kang et al. (2025). The individual environment scripts focus on GLADIUS reward recovery for CartPole and Acrobot separately, with evaluation against the true simulator rewards. A shared utility module handles data loading, state binning, encoder construction, and policy evaluation.

Scripts in this example directory:

- ``benchmark_gladius_vs_airl.py`` -- head-to-head GLADIUS versus AIRL benchmark on three Gym environments
- ``run_gladius_cartpole.py`` -- GLADIUS reward recovery on CartPole expert demonstrations
- ``run_gladius_acrobot.py`` -- GLADIUS reward recovery on Acrobot heuristic expert demonstrations
- ``gym_irl_utils.py`` -- shared utilities for data loading, state binning, and policy evaluation
