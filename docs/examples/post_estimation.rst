Rust Bus Post-Estimation Diagnostics
======================================

This example demonstrates the full post-estimation diagnostic suite on the Rust (1987) bus engine dataset. After estimating two models on the same data, the diagnostics compare fit, test hypotheses, measure reward equivalence, and probe identification strength. All 15 tools shown here are available from ``econirl.inference``.

Quick start
-----------

Fit NFXP and CCP on the bus engine data, then compare them side by side with ``etable``.

.. code-block:: python

   from econirl import NFXP, CCP
   from econirl.datasets import load_rust_bus
   from econirl.inference import etable

   df = load_rust_bus()

   nfxp = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")
   ccp  = CCP(discount=0.9999, num_policy_iterations=3).fit(df, state="mileage_bin", action="replaced", id="bus_id")

   print(etable(nfxp._result, ccp._result, model_names=["NFXP", "CCP"]))

.. code-block:: text

                                 NFXP           CCP
   ================================================
                theta_c      0.0010**    -0.0019***
                                (0.0004)      (0.0000)
                     RC     3.0724***     2.1062***
                                (0.0747)      (0.0000)
   ------------------------------------------------
           Observations         9,410         9,410
         Log-Likelihood     -1,900.33     -2,047.56
                    AIC       3,804.7       4,099.1
                    BIC       3,819.0       4,113.4
   ================================================
   Note: * p<0.10, ** p<0.05, *** p<0.01

The ``etable`` function produces a stargazer-style table with estimates, standard errors in parentheses, and significance stars. It also supports LaTeX and HTML output through the ``output`` argument.

Hypothesis tests
-----------------

The Vuong test compares two non-nested models by evaluating their per-observation log-likelihood differences. A positive z-statistic means the first model fits the data better.

.. code-block:: python

   from econirl.inference import vuong_test
   import jax.numpy as jnp

   obs_states = jnp.array(df["mileage_bin"].values, dtype=jnp.int32)
   obs_actions = jnp.array(df["replaced"].values, dtype=jnp.int32)

   vt = vuong_test(
       jnp.array(nfxp.policy_), jnp.array(ccp.policy_),
       obs_states, obs_actions,
   )
   print(f"z = {vt['statistic']:.4f}, p = {vt['p_value']:.4f}, direction = {vt['direction']}")

.. code-block:: text

   z = 8.8291, p = 0.0000, direction = model_1

With a z-statistic of 8.83 and a p-value below 0.001, the test strongly favors NFXP over CCP on this dataset.

The ``likelihood_ratio_test`` and ``score_test`` functions test nested restrictions. The LR test takes two fitted summaries and computes the statistic directly.

.. code-block:: python

   from econirl.inference import likelihood_ratio_test

   lr = likelihood_ratio_test(restricted=ccp._result, unrestricted=nfxp._result)
   print(f"LR = {lr['statistic']:.2f}, df = {lr['df']}, p = {lr['p_value']:.4f}")

Prediction quality
-------------------

Three metrics measure how well each model predicts the observed choices.

.. code-block:: python

   from econirl.inference import brier_score, kl_divergence, efron_pseudo_r_squared

   bs_nfxp = brier_score(jnp.array(nfxp.policy_), obs_states, obs_actions)
   bs_ccp  = brier_score(jnp.array(ccp.policy_), obs_states, obs_actions)
   print(f"Brier score NFXP: {bs_nfxp['brier_score']:.6f}")
   print(f"Brier score CCP:  {bs_ccp['brier_score']:.6f}")

.. code-block:: text

   Brier score NFXP: 0.097296
   Brier score CCP:  0.101451

.. code-block:: python

   kl_nfxp = kl_divergence(data_ccps, jnp.array(nfxp.policy_), state_freq)
   kl_ccp  = kl_divergence(data_ccps, jnp.array(ccp.policy_), state_freq)
   print(f"KL divergence NFXP: {kl_nfxp['kl_divergence']:.6f}")
   print(f"KL divergence CCP:  {kl_ccp['kl_divergence']:.6f}")

.. code-block:: text

   KL divergence NFXP: 0.004101
   KL divergence CCP:  0.019748

.. code-block:: python

   er_nfxp = efron_pseudo_r_squared(jnp.array(nfxp.policy_), obs_states, obs_actions)
   er_ccp  = efron_pseudo_r_squared(jnp.array(ccp.policy_), obs_states, obs_actions)
   print(f"Efron R² NFXP: {er_nfxp['efron_r_squared']:.4f}")
   print(f"Efron R² CCP:  {er_ccp['efron_r_squared']:.4f}")

.. code-block:: text

   Efron R² NFXP:  0.0009
   Efron R² CCP:  -0.0417

NFXP achieves a lower Brier score and KL divergence on every metric, consistent with the Vuong test result. The Efron R-squared values are small because the replacement event is rare, so even good models produce modest values in absolute terms.

.. list-table:: Prediction Metrics Summary
   :header-rows: 1

   * - Metric
     - NFXP
     - CCP
   * - Brier Score
     - 0.0973
     - 0.1015
   * - KL Divergence
     - 0.0041
     - 0.0197
   * - Efron R-squared
     - 0.0009
     - -0.0417

Specification tests
--------------------

The CCP consistency test compares the empirical choice probabilities to the model-implied probabilities using a Pearson chi-squared statistic.

.. code-block:: python

   from econirl.inference import ccp_consistency_test

   test = ccp_consistency_test(
       data_ccps, jnp.array(nfxp.policy_),
       jnp.array(state_counts),
       num_estimated_params=nfxp._result.num_parameters,
   )
   print(f"chi2 = {test['statistic']:.2f}, df = {test['df']}, p = {test['p_value']:.4f}")

.. code-block:: text

   chi2 = 68.81, df = 50, p = 0.0399

The NFXP model produces a chi-squared statistic of 68.81 with 50 degrees of freedom and a p-value of 0.04. The marginal rejection suggests a reasonable but imperfect fit, which is typical for structural models estimated on real data.

Reward comparison
------------------

The EPIC distance measures how different two reward functions are after removing potential-based shaping and scale differences. A distance of zero means the two rewards induce the same optimal policy for any transition dynamics.

.. code-block:: python

   from econirl.inference import epic_distance

   # Build reward vectors from estimated parameters
   reward_nfxp = np.zeros((num_states, 2))
   reward_nfxp[:, 0] = -nfxp.params_["theta_c"] * np.arange(num_states)
   reward_nfxp[:, 1] = -nfxp.params_["RC"]

   reward_ccp = np.zeros((num_states, 2))
   reward_ccp[:, 0] = -ccp.params_["theta_c"] * np.arange(num_states)
   reward_ccp[:, 1] = -ccp.params_["RC"]

   epic = epic_distance(jnp.array(reward_nfxp), jnp.array(reward_ccp), 0.9999)
   print(f"EPIC distance:       {epic['epic_distance']:.6f}")
   print(f"Pearson correlation: {epic['pearson_correlation']:.6f}")

.. code-block:: text

   EPIC distance:       0.021809
   Pearson correlation: 0.999049

The EPIC distance between the NFXP and CCP reward vectors is 0.022 with a Pearson correlation of 0.999. The two estimators recover nearly identical reward structures despite different computational approaches.

The shaping detection function tests whether two rewards differ only by a potential function and recovers the potential if they do. A synthetic example demonstrates exact recovery.

.. code-block:: python

   from econirl.inference import detect_reward_shaping

   r_base = jnp.zeros((5, 2, 5))
   phi_true = jnp.array([0.0, 1.5, -0.8, 2.3, -1.1])
   shaped = r_base + 0.99 * phi_true[None, None, :] - phi_true[:, None, None]

   result = detect_reward_shaping(r_base, shaped, 0.99)
   print(f"Is shaping:         {result['is_shaping']}")
   print(f"Relative residual:  {result['relative_residual']:.6f}")
   print(f"Recovered potential: {[f'{x:.2f}' for x in result['potential']]}")
   print(f"True potential:      {[f'{float(x):.2f}' for x in phi_true]}")

.. code-block:: text

   Is shaping:         True
   Relative residual:  0.000000
   Recovered potential: ['0.00', '1.50', '-0.80', '2.30', '-1.10']
   True potential:      ['0.00', '1.50', '-0.80', '2.30', '-1.10']

Bundled diagnostics
--------------------

The ``diagnostics()`` method on any ``EstimationSummary`` bundles goodness of fit, identification, numerical quality, and convergence information into a single dict.

.. code-block:: python

   diag = nfxp._result.diagnostics()
   print(diag.keys())
   print(f"AIC: {diag['goodness_of_fit']['aic']}")
   print(f"Identification: {diag['identification']['status']}")
   print(f"Converged: {diag['convergence']['converged']}")

.. code-block:: text

   dict_keys(['goodness_of_fit', 'identification', 'numerical_quality', 'convergence'])
   AIC: 3804.7
   Identification: Well-identified
   Converged: True

Reward identifiability
-----------------------

The ``check_reward_identifiability`` function implements the Kim, Garg, Shiragur, and Ermon (2021) graph-theoretic test. It builds the domain graph from the MDP transition support and checks aperiodicity. An aperiodic and strongly connected graph means the reward is uniquely recoverable under MaxEnt IRL.

.. code-block:: python

   from econirl.inference import check_reward_identifiability
   import numpy as np

   # Rust bus transitions are aperiodic (self-loops at every mileage bin)
   T = np.array(nfxp.transitions_)
   ident = check_reward_identifiability(T)
   print(f"Identifiable: {ident['is_identifiable']}")
   print(f"Period:       {ident['period']}")
   print(f"Status:       {ident['status']}")

.. code-block:: text

   Identifiable: True
   Period:       1
   Status:       Strongly identifiable: domain graph is aperiodic (period 1)

A periodic transition structure would fail this test. For example, a two-state chain that alternates deterministically between states has period 2 and is not identifiable.

Discount factor sensitivity
----------------------------

The discount factor cannot be separately identified from flow utilities without exclusion restrictions (Magnac and Thesmar, 2002). The ``discount_factor_sensitivity`` function re-estimates the model at a grid of beta values to reveal how parameters shift with the assumed discount factor.

.. code-block:: python

   from econirl.inference import discount_factor_sensitivity
   from econirl.estimation.nfxp import NFXPEstimator

   sens = discount_factor_sensitivity(
       estimator=NFXPEstimator(outer_max_iter=200, inner_max_iter=50000),
       panel=panel, utility=utility, problem=problem, transitions=transitions,
       beta_grid=[0.90, 0.95, 0.99, 0.999, 0.9999],
   )
   for i, beta in enumerate(sens["beta_values"]):
       params = sens["params"][i]
       ll = sens["log_likelihoods"][i]
       print(f"beta={beta:.4f}  theta_c={params.get('theta_c', 'N/A'):>10.6f}  "
             f"RC={params.get('RC', 'N/A'):>8.4f}  LL={ll:>10.2f}")

Large swings in the parameter estimates across the beta grid signal that the discount factor and the structural parameters are not separately identified. Stable estimates across the grid provide reassurance.

Warm-start bootstrap
---------------------

The ``warm_start_bootstrap`` function implements the Kasahara and Shimotsu (2008) procedure. It resamples individuals with replacement and runs only one Newton step from the MLE estimate per replicate, achieving the same asymptotic coverage as a full bootstrap at a fraction of the cost.

.. code-block:: python

   from econirl.inference import warm_start_bootstrap
   from econirl.estimation.nfxp import NFXPEstimator

   boot = warm_start_bootstrap(
       estimator=NFXPEstimator(outer_max_iter=200, inner_max_iter=50000),
       panel=panel, utility=utility, problem=problem, transitions=transitions,
       mle_result=nfxp._result,
       n_bootstrap=499, n_newton_steps=1, seed=42,
   )
   print(f"Bootstrap SEs:  {boot['standard_errors']}")
   print(f"Successful:     {boot['n_successful']} / 499")

The bootstrap standard errors should be close to the Hessian-based asymptotic standard errors. When they differ substantially, the likelihood surface may be non-quadratic and the asymptotic approximation unreliable.

Profile likelihood
-------------------

The ``profile_likelihood`` function fixes one parameter at a grid of values and re-optimizes the remaining parameters at each point. A sharply peaked profile means the parameter is well-identified. A flat profile signals weak identification. Profile confidence intervals invert the chi-squared criterion and are more reliable than Wald intervals when the likelihood is non-quadratic.

.. code-block:: python

   from econirl.inference import profile_likelihood
   from econirl.estimation.nfxp import NFXPEstimator

   prof = profile_likelihood(
       estimator=NFXPEstimator(outer_max_iter=200, inner_max_iter=50000),
       panel=panel, utility=utility, problem=problem, transitions=transitions,
       mle_result=nfxp._result,
       param_index=0,  # theta_c
       n_points=15,
   )
   print(f"Parameter:   {prof['param_name']}")
   print(f"MLE value:   {prof['mle_value']:.6f}")
   print(f"Profile CI:  [{prof['ci_lower']:.6f}, {prof['ci_upper']:.6f}]")

The profile CI may be asymmetric, unlike the symmetric Wald interval. This asymmetry is informative because it reflects the actual curvature of the likelihood surface rather than assuming a quadratic approximation.

Running the example
--------------------

.. code-block:: bash

   python examples/post_estimation_diagnostics.py

Reference
---------

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

Vuong, Q. H. (1989). Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses. Econometrica, 57(2), 307-333.

Gleave, A., Dennis, M., Legg, S., Russell, S., and Leike, J. (2020). Quantifying Differences in Reward Functions. ICLR 2021.

Kim, K., Garg, S., Shiragur, K., and Ermon, S. (2021). Reward Identification in Inverse Reinforcement Learning. ICML, PMLR 139, 5496-5505.

Kasahara, H. and Shimotsu, K. (2008). Pseudo-likelihood Estimation and Bootstrap Inference for Structural Discrete Markov Decision Models. Journal of Econometrics, 146(1), 92-106.

Magnac, T. and Thesmar, D. (2002). Identifying Dynamic Discrete Decision Processes. Econometrica, 70(2), 801-816.
