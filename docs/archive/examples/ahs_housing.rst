AHS Housing Mobility
====================

This example estimates a household mobility model with known ground truth reward parameters. The transition dynamics and initial state distribution are calibrated from the 2023 American Housing Survey, a national cross-section of 55,000 households. Because the AHS is a single cross-section rather than a panel, there are no observed individual trajectories. Instead, the example defines ground truth structural parameters, computes the optimal policy via soft value iteration, and generates a synthetic panel from that policy. Four estimators then recover the parameters, demonstrating both exact structural methods and neural approximation methods on a realistic housing domain.

The state space has 54 states arranged as 2 tenure types (own or rent) by 3 age bins (young, middle, senior) by 3 income bins (low, middle, high) by 3 duration bins (new, established, long-tenure). The agent decides each period whether to stay in the current unit or move. Moving resets duration to zero, may change tenure type with 15 percent probability, and redraws income from the cross-sectional distribution. Staying increments duration and allows slow age and income transitions.

Semi-synthetic methodology
--------------------------

The transitions are structural rather than empirical. Duration increments deterministically for stayers and resets for movers. Age and income follow slow random walks. These transition probabilities are calibrated to match the cross-sectional duration distribution and move rates from the AHS. The feature matrix uses median housing burden values computed from the AHS cross-section. The initial state distribution is the empirical distribution of states in the cross-section.

The ground truth parameters define the flow utility of staying and the cost of moving. The soft value iteration operator computes the optimal stochastic policy under these parameters, and the ``simulate_panel_from_policy`` function generates 5,000 trajectories of 20 periods each (100,000 observations).

.. code-block:: python

   from econirl.core.bellman import SoftBellmanOperator
   from econirl.core.solvers import value_iteration
   from econirl.simulation.synthetic import simulate_panel_from_policy

   operator = SoftBellmanOperator(problem, transitions)
   true_utility = utility.compute(true_params)
   solver_result = value_iteration(operator, true_utility, tol=1e-10)

   panel = simulate_panel_from_policy(
       problem=problem, transitions=transitions,
       policy=solver_result.policy,
       initial_distribution=initial_dist,
       n_individuals=5000, n_periods=20, seed=42,
   )

Parameter recovery
------------------

Four estimators recover the structural parameters from the generated panel. NFXP solves the Bellman fixed point exactly. CCP uses Hotz-Miller inversion with 10 NPL refinement iterations. NNES trains a neural V-network to approximate the value function (Nguyen 2025). TD-CCP uses the semigradient algorithm with cross-fitting (Adusumilli and Eckardt 2025).

.. list-table:: Parameter Recovery (5000 households, 100K observations)
   :header-rows: 1

   * - Parameter
     - True
     - NFXP
     - CCP
     - NNES
     - TD-CCP
   * - theta_burden
     - -0.50
     - -0.460
     - -0.460
     - 0.085
     - -1.218
   * - theta_duration
     - 0.30
     - 0.291
     - 0.291
     - 0.595
     - 0.491
   * - theta_renter
     - -0.40
     - -0.417
     - -0.417
     - -3.099
     - -0.544
   * - theta_age
     - 0.60
     - 0.597
     - 0.597
     - 0.528
     - 0.181
   * - move_cost
     - -1.50
     - -1.535
     - -1.535
     - -3.053
     - -2.229

NFXP and CCP produce identical estimates, as expected when NPL converges to the MLE. Both recover all five parameters with the correct signs and magnitudes within 10 percent of the true values. The housing burden coefficient is negative 0.460 versus the true negative 0.50, meaning that the model correctly identifies that high cost-to-income ratios push households toward moving. The age coefficient is 0.597 versus 0.60, confirming that older households strongly prefer staying.

NNES overestimates the renter effect and the move cost by roughly a factor of two, and it flips the sign on the burden parameter. This is consistent with the NNES V-network absorbing part of the state-dependent variation into the flexible approximation, distorting the structural parameter estimates. On a 54-state problem where exact Bellman solution is feasible, the structural methods outperform the neural approximation.

TD-CCP recovers correct signs for all parameters except theta_age, which it underestimates at 0.181. The semigradient basis with 8 dimensions cannot fully capture the value function in a 54-state space with heterogeneous transition dynamics across tenure types.

Running the example
-------------------

.. code-block:: bash

   python examples/ahs-housing/run_ccp.py

Reference
---------

American Housing Survey 2023 National. U.S. Census Bureau and U.S. Department of Housing and Urban Development.

Ferreira, F., Gyourko, J., Tracy, J. (2010). Housing Busts and Household Mobility. Journal of Urban Economics 68(1): 34-45.
