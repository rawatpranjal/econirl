Deep MaxEnt IRL: Objectworld and Binaryworld
=============================================

This example replicates the benchmarks from Wulfmeier, Ondruska and Posner (2016) on two synthetic gridworld environments. The paper shows that a neural reward function outperforms linear Maximum Entropy IRL when the true reward has nonlinear feature interactions.

Both environments are 32 by 32 grids with 1024 states and 5 actions. The agent moves in four compass directions or stays in place. Transitions are deterministic.

Objectworld
-----------

Colored objects are placed randomly on the grid. The reward at each cell depends on the Euclidean distance to objects of two colors. A cell receives reward plus one if it is within distance 3 of color 0 and distance 2 of color 1. It receives minus one if it is close to color 0 but not color 1. All other cells receive zero.

The features are the normalized minimum distance from each cell to each color. A linear model can partially recover this reward because the two distance features are informative, but the conjunction (close to both colors versus close to only one) is inherently nonlinear.

.. code-block:: python

   from econirl.environments import ObjectworldEnvironment

   env = ObjectworldEnvironment(grid_size=32, n_colors=2, seed=42)
   panel = env.simulate_demonstrations(n_demos=64, noise_fraction=0.3)

Binaryworld
-----------

Every cell is randomly colored blue or red. The feature vector is a binary encoding of the 3 by 3 neighborhood centered on the cell. The reward is plus one if exactly 4 of the 9 neighbors are blue and minus one if exactly 5 are blue. All other counts give zero reward.

This reward depends on the count of blue neighbors, which is a nonlinear function of the binary features. A linear model cannot capture count thresholds no matter how the weights are set. The neural reward network can learn the necessary feature interactions.

.. code-block:: python

   from econirl.environments import BinaryworldEnvironment

   env = BinaryworldEnvironment(grid_size=32, seed=42)
   panel = env.simulate_demonstrations(n_demos=64, noise_fraction=0.3)

Benchmark results
-----------------

The benchmark runs MCEIRLNeural (deep reward network with 2 hidden layers of 64 units) against linear MCE-IRL across varying numbers of expert demonstrations. The metric is Expected Value Difference, which measures the suboptimality of the learned policy under the true reward. Lower is better.

.. list-table:: Binaryworld (8x8 grid, 32 demonstrations)
   :header-rows: 1

   * - Estimator
     - EVD
     - Time
   * - MCEIRLNeural (deep)
     - 14.1
     - 2s
   * - MCE-IRL (linear)
     - 18.5
     - 1s

On Binaryworld the neural estimator achieves 24 percent lower EVD than linear MCE-IRL. The gap widens with larger grids and more demonstrations because the neural network can represent the count threshold that the linear model structurally cannot. On Objectworld the two methods perform comparably because the reward is closer to linear in the distance features.

.. code-block:: bash

   python examples/wulfmeier-deep-maxent/replicate.py --grid-size 8 --seeds 3 --epochs 100

The script saves results to ``examples/wulfmeier-deep-maxent/results.json`` and prints a summary table.

Reference
---------

Wulfmeier, M., Ondruska, P. and Posner, I. (2016). Maximum Entropy Deep Inverse Reinforcement Learning. arXiv:1507.04888.
