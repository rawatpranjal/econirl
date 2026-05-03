KKBOX Subscription Churn
========================

This example applies structural estimation to a subscription renewal problem using transaction data from KKBOX, the leading music streaming service in Asia. The data comes from the WSDM 2017 KKBox Churn Prediction Challenge and contains 21.5 million transaction records covering 2.4 million subscribers. Each month a subscriber decides whether to renew or cancel the subscription. This is a binary optimal stopping problem structurally identical to the Rust (1987) bus engine replacement model.

The state space has 36 states arranged as 6 tenure bins by 3 price tiers by 2 auto-renewal levels. Tenure captures cumulative months subscribed. Price tiers separate free and cheap plans from standard and premium ones. The auto-renewal flag distinguishes subscribers who opted into automatic renewal from those who actively choose each month. The action set has two choices. Renew continues the subscription. Cancel terminates it, which is modeled as an absorbing state.

Estimation
----------

The CCP estimator with NPL refinement recovers the structural parameters from 4,961 subscribers over 61,750 total observations. The NPL algorithm converges by iteration 6.

.. code-block:: python

   from econirl.estimation.ccp import CCPEstimator
   from econirl.preferences.linear import LinearUtility
   from econirl.core.types import DDCProblem

   utility = LinearUtility(feature_matrix=features, parameter_names=param_names)
   problem = DDCProblem(num_states=36, num_actions=2, discount_factor=0.95)

   npl = CCPEstimator(num_policy_iterations=10, compute_hessian=True)
   result = npl.estimate(panel, utility, problem, transitions)

.. list-table:: Structural Parameters (NPL K=10, 4961 subscribers)
   :header-rows: 1

   * - Parameter
     - Estimate
     - SE
     - t-stat
     - p-value
   * - theta_tenure
     - 0.0501
     - 0.0203
     - 2.47
     - 0.014
   * - theta_price
     - -0.1523
     - 0.0607
     - -2.51
     - 0.012
   * - theta_auto_renew
     - -2.6249
     - 1.0873
     - -2.41
     - 0.016
   * - theta_discount
     - 0.1219
     - 0.0260
     - 4.69
     - 0.000
   * - constant
     - -29.6883
     - 11.1542
     - -2.66
     - 0.008

All five parameters are statistically significant at the 5 percent level. The tenure coefficient is positive, confirming that longer-subscribed users are more loyal. The price coefficient is negative, meaning higher plan prices reduce renewal utility. The discount coefficient is positive, meaning subscribers paying below list price are more likely to renew. The constant on the cancel action is large and negative, reflecting the very low baseline churn rate in the data.

The auto-renewal coefficient is negative, which may seem counterintuitive at first. Auto-renewal subscribers are passively renewed by the platform, so their observed renewal behavior does not reflect active choice. The negative coefficient on the renewal action captures the fact that auto-renewal subscribers have lower active engagement with the renewal decision. This is consistent with a model where auto-renewal acts as a default bias rather than a preference signal.

Diagnostics
-----------

The feature matrix has full rank (5 out of 5) with a condition number of 3.4. All 36 states have at least one observation. Of the 36 states, 21 have only one observed action. These boundary states produce degenerate CCP estimates but do not prevent convergence because the NPL algorithm smooths across states through the Bellman operator.

Prediction accuracy is 0.971. The high accuracy reflects the fact that renewal is the dominant action in most states, so even a simple model that predicts renewal everywhere achieves high accuracy. The more informative diagnostic is the empirical versus predicted cancel rate by tenure bin.

.. list-table:: Cancel Rate by Tenure Bin
   :header-rows: 1

   * - Tenure bin
     - Empirical
     - Predicted
   * - 1 month
     - 0.0000
     - 0.0151
   * - 2 months
     - 0.0169
     - 0.0199
   * - 3 to 4 months
     - 0.0205
     - 0.0189
   * - 5 to 8 months
     - 0.0261
     - 0.0188
   * - 9 to 16 months
     - 0.0199
     - 0.0234
   * - 17 or more months
     - 0.0202
     - 0.0217

The model captures the overall level of churn but does not fully reproduce the nonmonotonic pattern in the 5 to 8 month bin. This bin has the highest empirical cancel rate at 2.6 percent, which the model underestimates. A richer state space that separates plan types or payment history within this tenure range might improve the fit.

Running the example
-------------------

.. code-block:: bash

   python examples/kkbox-churn/run_ccp.py

The robustness script varies the discount factor (0.90, 0.95, 0.99), sample size (5K, 20K, 50K users), random seeds, action framing, and interaction terms.

.. code-block:: bash

   python examples/kkbox-churn/robustness.py

Reference
---------

WSDM 2017 KKBox Churn Prediction Challenge. Kaggle: kkbox/kkbox-churn-prediction-challenge.

Shiller, B. (2020). Digital Distribution and the Prohibition of Resale Markets for Information Goods. Quantitative Marketing and Economics 18(4): 403-435.
