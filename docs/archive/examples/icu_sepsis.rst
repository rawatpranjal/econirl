ICU Sepsis Treatment
====================

.. image:: /_static/mdp_schematic_icu_sepsis.png
   :alt: ICU sepsis MDP structure showing patient state, 5 by 5 dosing action grid, and absorbing discharge or death states.
   :width: 80%
   :align: center

This example applies inverse reinforcement learning to the ICU-Sepsis benchmark MDP derived from MIMIC-III patient records. The data comes from Komorowski et al. (2018) and was packaged as a standalone benchmark by Killian et al. (2024). A clinician decides at each four-hour window how much IV fluid and vasopressor to administer to a sepsis patient, balancing the benefits of aggressive treatment against the risks of fluid overload and vasoconstriction.

.. image:: /_static/icu_sepsis_overview.png
   :alt: ICU-Sepsis benchmark MDP overview showing illness severity distribution across 713 clinical states, clinician treatment intensity by SOFA quintile, and admission severity distribution.
   :width: 100%

The MDP has 716 states representing clusters of patient physiology (vital signs, lab values, demographics), 25 actions arranged as a 5 by 5 grid of IV fluid dose and vasopressor dose, and transition probabilities estimated from real patient trajectories. The reward is plus one upon patient survival and zero otherwise. The expert policy is the aggregate clinician behavior observed in MIMIC-III.

Quick start
-----------

Load the environment and generate expert demonstrations from the MIMIC-III clinician policy.

.. code-block:: python

   from econirl.environments.icu_sepsis import ICUSepsisEnvironment
   from econirl.datasets.icu_sepsis import load_icu_sepsis, load_icu_sepsis_mdp

   env = ICUSepsisEnvironment(discount_factor=0.99)
   panel = load_icu_sepsis(n_individuals=2000, max_steps=20, as_panel=True)

The raw MDP components are available for direct analysis.

.. code-block:: python

   mdp = load_icu_sepsis_mdp()
   print(mdp["transitions"].shape)      # (25, 716, 716)
   print(mdp["expert_policy"].shape)     # (716, 25)
   print(mdp["sofa_scores"][:5])         # SOFA scores for first 5 states

Estimation
----------

NFXP and CCP recover the implicit reward function from 1600 training patients (14512 observations). The linear utility model has four features: SOFA score (illness severity), IV fluid dose, vasopressor dose, and an absorbing state indicator.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.preferences.linear import LinearUtility

   utility = LinearUtility(feature_matrix=env.feature_matrix, parameter_names=env.parameter_names)
   nfxp = NFXPEstimator()
   result = nfxp.estimate(panel, utility, env.problem_spec, env.transition_matrices)

.. list-table:: Estimated Parameters (2000 patients, 20 steps)
   :header-rows: 1

   * - Parameter
     - NFXP
     - CCP
     - Std Error (NFXP)
   * - sofa_weight
     - 1.1288
     - -0.0028
     - 0.2465
   * - fluid_weight
     - -0.3270
     - -0.3508
     - 0.0255
   * - vaso_weight
     - -4.1528
     - -4.0477
     - 0.0305
   * - absorbing_weight
     - 0.4291
     - -0.1435
     - 0.0888

The large negative vasopressor weight (negative 4.15) reveals that clinicians strongly penalize aggressive vasopressor dosing. The fluid weight is also negative but much smaller in magnitude (negative 0.33), indicating that fluid overload risk is a concern but less dominant than vasopressor side effects. Both estimators agree on these signs and relative magnitudes.

Post-estimation diagnostics
---------------------------

The ``etable`` function compares NFXP and CCP side by side with standard errors and significance stars.

.. code-block:: python

   from econirl.inference import etable
   print(etable(nfxp_result, ccp_result))

NFXP and CCP agree on the sign and magnitude of the vasopressor and fluid weights but diverge on the SOFA and absorbing weights. The SOFA weight is 1.13 under NFXP but negative 0.003 under CCP. This divergence likely reflects the sensitivity of the absorbing state structure to the Hotz-Miller inversion. Absorbing states (death and discharge) concentrate the data distribution and can distort the CCP likelihood surface, similar to the FrozenLake identification challenge.

The Brier score measures overall prediction quality. On 716 states with 25 actions, the Brier score is dominated by the many low-probability actions that the model correctly assigns near-zero probability.

.. code-block:: python

   from econirl.inference import brier_score
   bs = brier_score(nfxp_result.policy, obs_states, obs_actions)

Counterfactual analysis
-----------------------

Doubling the vasopressor cost simulates a policy intervention that restricts aggressive vasopressor dosing. The counterfactual shifts clinician behavior strongly toward lower vasopressor levels.

.. list-table:: Vasopressor Distribution Under Counterfactual
   :header-rows: 1

   * - Vasopressor Level
     - Baseline
     - Counterfactual
     - Change
   * - Level 0 (none)
     - 0.649
     - 0.874
     - +0.226
   * - Level 1
     - 0.230
     - 0.110
     - -0.121
   * - Level 2
     - 0.082
     - 0.014
     - -0.068
   * - Level 3
     - 0.029
     - 0.002
     - -0.027
   * - Level 4 (max)
     - 0.010
     - 0.000
     - -0.010

The welfare change is negative 29.8, reflecting the cost of restricting treatment options. Under the counterfactual, 87 percent of treatment decisions use no vasopressors at all, compared to 65 percent at baseline.

Elasticity analysis shows that the SOFA weight has a modest effect on welfare and nearly no effect on the policy distribution. The treatment choice is dominated by the drug cost parameters, not the severity indicator.

.. list-table:: SOFA Weight Elasticity
   :header-rows: 1

   * - % Change
     - Welfare Change
     - Avg Policy Change
   * - -50%
     - -1.749
     - 0.001
   * - -25%
     - -0.882
     - 0.001
   * - +25%
     - +0.869
     - 0.001
   * - +50%
     - +1.753
     - 0.001

Run the full example
--------------------

.. code-block:: bash

   python examples/icu-sepsis/run_estimation.py

This script generates 2000 patient trajectories from the clinician policy, estimates with NFXP and CCP, and runs counterfactual and elasticity analysis.

References
----------

Komorowski, M., Celi, L.A., Badawi, O., Gordon, A.C., and Faisal, A.A. (2018). The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. Nature Medicine, 24, 1716 to 1720.

Killian, T.W., Shan, J., Krishnamurthy, K., Joshi, P., Srinivasan, A., Lam, J., and Celi, L.A. (2024). ICU-Sepsis: A Benchmark MDP Built from Real Medical Data. NeurIPS Datasets and Benchmarks.
