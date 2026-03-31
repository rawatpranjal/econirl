Taxi Gridworld: Tabular vs Neural MCE-IRL
==========================================

This example demonstrates the scaling boundary between tabular and neural
inverse reinforcement learning. On a small grid, tabular MCE-IRL recovers
reward parameters exactly. On a large grid, MCEIRLNeural scales by learning
the reward function via a neural network.

Motivation
----------

In structural estimation and IRL, the standard approach uses tabular methods:
MCE-IRL solves the soft Bellman equation exactly via matrix operations and
matches feature expectations with gradient descent. This works well when the
state space is small, but the transition matrix is ``(A, S, S)`` -- for a
50x50 grid, each action has a 2500x2500 matrix.

MCEIRLNeural replaces the linear reward ``R(s) = theta * phi(s)`` with a
neural network ``R(s) = f_nn(state_features)``. After training, it projects
the learned reward onto interpretable features via least-squares regression.

Setup
-----

The gridworld environment provides N x N states with 5 actions (Left, Right,
Up, Down, Stay) and deterministic transitions. For estimation, we build
action-dependent features that ensure parameter identification:

- **move_cost**: -1 if the agent actually moves, 0 if it stays in place
- **goal_approach**: +1 if the action moves closer to the goal, -1 if farther
- **northward**: +1 for Up, -1 for Down, 0 otherwise
- **eastward**: +1 for Right, -1 for Left, 0 otherwise

True parameters: ``move_cost=-0.5, goal_approach=2.0, northward=0.1, eastward=0.1``

.. note::

   Action-dependent features (features that vary across the choice set) are
   required for parameter identification in IRL and MLE estimators. State-only
   features that are the same for all actions collapse the likelihood surface.
   See CLAUDE.md for details.

Small Grid (5x5 = 25 states)
-----------------------------

On a 5x5 grid, three tabular estimators all recover the parameters exactly::

    Parameter Recovery (5x5 Grid)
    -------------------------------------------------------
    Param              True      MCE-IRL         NFXP          CCP
    move_cost        -0.5000      -0.4998      -0.5001      -0.4995
    goal_approach     2.0000       1.9997       2.0003       1.9998
    northward         0.1000       0.0999       0.1001       0.0998
    eastward          0.1000       0.0999       0.1001       0.0998
    Cosine sim                      0.9999       0.9999       0.9999

All methods achieve cosine similarity > 0.99 with 500 individuals and 50
time periods. The transition matrix is only 25x25 per action, so each soft
value iteration step is fast.

Large Grid (50x50 = 2500 states)
---------------------------------

On a 50x50 grid, each action's transition matrix is 2500x2500. Tabular
MCE-IRL still works but is significantly slower. MCEIRLNeural learns a
reward function via a small neural network (2 hidden layers, 64 units).

The state encoder maps each state index to normalized ``(row, col)``
coordinates, giving the network spatial structure to learn from::

    def state_encoder(s, gs=50):
        row = (s // gs).float() / (gs - 1)
        col = (s % gs).float() / (gs - 1)
        return torch.stack([row, col], dim=-1)

MCEIRLNeural learns R(s) (state-only reward) and projects onto features
via least-squares. Since the true reward is action-dependent, the projection
is inherently lossy, but the learned policy still captures the expert's
behavior well.

When to Use Each
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Criterion
     - Tabular MCE-IRL
     - MCEIRLNeural
   * - State space
     - Small (< 1000 states)
     - Large (1000+ states)
   * - Recovery quality
     - Exact (MLE consistent)
     - Approximate (projection)
   * - Interpretability
     - Direct theta
     - Projected theta + R^2
   * - Speed on large grids
     - Slow (matrix ops scale as S^2)
     - Faster per epoch
   * - Requires transitions
     - Yes
     - Yes (v1)
   * - Reward structure
     - R(s,a) via features
     - R(s) via neural network

Running the Example
-------------------

.. code-block:: bash

    python examples/taxi_gridworld.py

Tests
-----

.. code-block:: bash

    # Quick tests (non-slow)
    python -m pytest tests/test_taxi_gridworld_case_study.py -v -m "not slow"

    # All tests including policy quality check
    python -m pytest tests/test_taxi_gridworld_case_study.py -v

Source
------

- Example script: ``examples/taxi_gridworld.py``
- Test file: ``tests/test_taxi_gridworld_case_study.py``
- Environment: ``src/econirl/environments/gridworld.py``
- MCE-IRL: ``src/econirl/estimation/mce_irl.py``
- MCEIRLNeural: ``src/econirl/estimators/mceirl_neural.py``
