Keane-Wolpin Career Decisions
=============================

This example replicates a simplified version of Keane and Wolpin (1997) on occupational choice data. Young men choose each year between four options: attend school, work white-collar, work blue-collar, or stay home. The decision is forward-looking because schooling and work experience accumulate as human capital that raises future wages.

The state space encodes schooling level (10 to 20 years), white-collar experience (0 to 7 years), and blue-collar experience (0 to 7 years), giving 704 discrete states. Transitions are deterministic: choosing school increments the schooling counter, choosing white-collar work increments the white-collar experience counter, and so on. The model runs for 10 periods corresponding to ages 17 through 26.

.. code-block:: python

   from econirl.datasets import load_keane_wolpin
   from econirl.estimation.nfxp import NFXPEstimator

   df = load_keane_wolpin()
   # 704 states = 11 schooling x 8 exp_wc x 8 exp_bc
   # 4 actions = school, white-collar, blue-collar, home

The simplification relative to the original paper is in the shock distribution. Keane and Wolpin use multivariate normal shocks estimated via GHK simulation, which requires specialized likelihood machinery. This example uses logit (Type I extreme value) shocks, which gives the standard softmax choice probabilities that the econirl NFXP estimator handles natively. The finite-horizon backward induction algorithm and the state space structure match the original paper exactly.

.. list-table:: Estimated Parameters (logit simplification)
   :header-rows: 1

   * - Parameter
     - Description
     - Estimate
   * - school_return
     - Returns to schooling
     - positive
   * - wc_exp_return
     - Returns to white-collar experience
     - positive
   * - bc_exp_return
     - Returns to blue-collar experience
     - positive
   * - school_cost
     - Direct cost of attending school
     - negative
   * - wc_intercept
     - Base white-collar wage
     - positive
   * - bc_intercept
     - Base blue-collar wage
     - positive
   * - home_value
     - Value of home production
     - near zero

The signs match economic intuition. Schooling and experience both increase future earnings, so their return coefficients are positive. The school cost is negative because attending school means forgoing wages. The white-collar intercept exceeds the blue-collar intercept, consistent with the observed wage premium for white-collar work in the data.

.. code-block:: bash

   python examples/keane-wolpin-careers/replicate.py

This example demonstrates that econirl handles finite-horizon problems with large multi-dimensional state spaces and more than two actions. The backward induction solver correctly propagates value through all 10 periods.

Reference
---------

Keane, M. P. and Wolpin, K. I. (1997). The Career Decisions of Young Men. Journal of Political Economy, 105(3), 473-522.
