Freddie Mac Mortgage Repayment
==============================

This example estimates a mortgage repayment model with known ground truth reward parameters. The calibration data comes from the Freddie Mac Single-Family Loan-Level Dataset, which contains 148,938 loans with credit scores, loan-to-value ratios, delinquency histories, and repayment outcomes. Because the dataset provides origination-level summaries rather than monthly payment panels, the example defines ground truth structural parameters, computes the optimal borrower policy via soft value iteration, and generates a synthetic monthly panel. Four estimators then recover the parameters, demonstrating both structural and neural methods on a three-action mortgage termination problem.

The state space has 24 states arranged as 4 delinquency levels (current, 30-day, 60-day, 90-day or more) by 2 credit bins (below or above 660) by 3 loan-to-value bins (low, medium, high). Each month the borrower chooses one of three actions. Pay continues the mortgage. Prepay terminates the loan early through refinancing or sale. Default terminates the loan through foreclosure. Both prepay and default are absorbing.

Semi-synthetic methodology
--------------------------

The transition dynamics for the pay action model delinquency escalation and cure rates. A current borrower has a 5 percent chance of becoming 30 days delinquent. A 30-day delinquent borrower has a 30 percent cure rate and a 40 percent escalation rate. LTV amortizes slowly at 2 percent per period. Credit bin is fixed as a time-invariant borrower characteristic. The initial state distribution is calibrated from the cross-sectional distribution of credit scores and LTV ratios in the Freddie Mac data. All loans start current.

The monthly discount factor is 0.99, which implies an annual discount factor of approximately 0.886. Policy iteration with matrix evaluation computes the optimal stochastic policy, and ``simulate_panel_from_policy`` generates 10,000 loan trajectories of 60 months each (600,000 observations).

.. code-block:: python

   from econirl.core.bellman import SoftBellmanOperator
   from econirl.core.solvers import policy_iteration
   from econirl.simulation.synthetic import simulate_panel_from_policy

   operator = SoftBellmanOperator(problem, transitions)
   true_utility = utility.compute(true_params)
   solver_result = policy_iteration(
       operator, true_utility, tol=1e-10, max_iter=200, eval_method="matrix"
   )

   panel = simulate_panel_from_policy(
       problem=problem, transitions=transitions,
       policy=solver_result.policy,
       initial_distribution=initial_dist,
       n_individuals=10000, n_periods=60, seed=42,
   )

Parameter recovery
------------------

Four estimators recover the six structural parameters. NFXP solves the Bellman fixed point exactly. CCP uses Hotz-Miller inversion with 10 NPL iterations. NNES trains a neural V-network (Nguyen 2025). TD-CCP uses the semigradient algorithm with cross-fitting (Adusumilli and Eckardt 2025).

.. list-table:: Parameter Recovery (10K loans, 600K observations)
   :header-rows: 1

   * - Parameter
     - True
     - NFXP
     - CCP
     - NNES
     - TD-CCP
   * - theta_delinquency
     - -0.80
     - -0.810
     - -0.810
     - -5.535
     - -1.008
   * - theta_credit
     - 0.50
     - 0.485
     - 0.485
     - 0.747
     - 1.188
   * - theta_ltv
     - -0.40
     - -0.382
     - -0.382
     - -1.885
     - -1.613
   * - theta_equity
     - 0.60
     - 0.665
     - 0.664
     - 0.648
     - 0.730
   * - prepay_cost
     - -1.00
     - -1.066
     - -1.064
     - -0.978
     - -1.125
   * - default_cost
     - -2.00
     - -2.003
     - -2.003
     - -1.933
     - -1.998

NFXP and CCP produce nearly identical estimates, confirming that NPL converges to the MLE on this problem. Both recover all six parameters with correct signs. The default cost is estimated at negative 2.003 versus the true negative 2.0, accurate to within 0.2 percent. The delinquency coefficient is negative 0.810 versus the true negative 0.80, confirming that delinquency distress reduces the utility of continuing to pay. The equity coefficient on the prepay action is 0.665 versus the true 0.60, meaning the model correctly identifies that positive equity drives prepayment.

NNES overestimates the delinquency effect by a factor of seven (negative 5.535 versus negative 0.80) and the LTV effect by a factor of five. The neural V-network absorbs the state-dependent variation into the flexible approximation, distorting the structural parameters on the pay action. However, NNES recovers the action-specific intercepts accurately. The prepay cost (negative 0.978) and default cost (negative 1.933) are within 7 percent of their true values. This is because action-specific intercepts are identified directly from the observed action frequencies, independent of the V-network approximation.

TD-CCP recovers correct signs for all parameters and is particularly accurate on the default cost (negative 1.998 versus negative 2.0). The LTV and credit parameters show larger deviations, reflecting the difficulty of capturing the value function for a 3-action problem with an 8-dimensional semigradient basis.

Running the example
-------------------

.. code-block:: bash

   python examples/freddie-mac/run_ccp.py

Reference
---------

Freddie Mac Single-Family Loan-Level Dataset. http://www.freddiemac.com/research/datasets/sf-loanlevel-dataset.page

Campbell, J.Y., Cocco, J.F. (2015). A Model of Mortgage Default. Journal of Finance 70(4): 1495-1554.
