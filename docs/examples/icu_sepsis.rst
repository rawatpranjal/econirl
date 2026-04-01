ICU Sepsis Treatment
====================

.. image:: /_static/icu_sepsis_overview.png
   :alt: ICU-Sepsis benchmark MDP overview showing illness severity distribution across 713 clinical states, clinician treatment intensity by SOFA quintile, and admission severity distribution.
   :width: 100%

This example applies inverse reinforcement learning to the ICU-Sepsis benchmark MDP derived from MIMIC-III patient records. The data comes from Komorowski et al. (2018) and was packaged as a standalone benchmark by Killian et al. (2024). A clinician decides at each four-hour window how much IV fluid and vasopressor to administer to a sepsis patient, balancing the benefits of aggressive treatment against the risks of fluid overload and vasoconstriction.

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

Three estimators recover the implicit reward function from clinician behavior. The linear utility model has four features: SOFA score (illness severity), IV fluid dose, vasopressor dose, and an absorbing state indicator.

.. code-block:: python

   from econirl.estimation.nfxp import NFXPEstimator
   from econirl.estimation.ccp import CCPEstimator
   from econirl.estimation.mce_irl import MCEIRLEstimator
   from econirl.preferences.linear import LinearUtility
   import jax.numpy as jnp

   utility = LinearUtility(features=env.feature_matrix, parameter_names=env.parameter_names)
   init = jnp.zeros(4)

   nfxp = NFXPEstimator(env.problem_spec, env.transition_matrices)
   nfxp_result = nfxp.fit(panel, utility, init_params=init)

   ccp = CCPEstimator(env.problem_spec, env.transition_matrices)
   ccp_result = ccp.fit(panel, utility, init_params=init)

   mce = MCEIRLEstimator(env.problem_spec, env.transition_matrices)
   mce_result = mce.fit(panel, utility, init_params=init)

Clinical interpretation
-----------------------

The recovered parameters reveal how clinicians weigh severity against treatment intensity. A negative SOFA weight means sicker patients receive lower flow utility, consistent with the clinical reality that high-SOFA patients have worse outcomes regardless of treatment. Negative fluid and vasopressor weights reflect that clinicians implicitly penalize aggressive dosing, consistent with evidence on fluid overload and vasopressor-induced organ damage.

Since this is real clinical data with no known ground-truth reward, the parameters should be interpreted as the implicit preference structure that rationalizes observed behavior. Different estimators may recover different parameter magnitudes, but the signs and relative magnitudes should be consistent.

Run the full example
--------------------

.. code-block:: bash

   python examples/icu-sepsis/run_estimation.py

This script generates 2000 patient trajectories from the clinician policy, estimates with all three methods, and reports out-of-sample log-likelihood and accuracy on a held-out test set.

References
----------

Komorowski, M., Celi, L.A., Badawi, O., Gordon, A.C., and Faisal, A.A. (2018). The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. Nature Medicine, 24, 1716 to 1720.

Killian, T.W., Shan, J., Krishnamurthy, K., Joshi, P., Srinivasan, A., Lam, J., and Celi, L.A. (2024). ICU-Sepsis: A Benchmark MDP Built from Real Medical Data. NeurIPS Datasets and Benchmarks.
