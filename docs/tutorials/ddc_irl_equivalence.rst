DDC and IRL Equivalence
=======================

Dynamic Discrete Choice estimation via NFXP and Maximum Entropy Inverse Reinforcement Learning optimize the same maximum likelihood objective. This tutorial proves the equivalence on a minimal example, applies MCE IRL and Max Margin IRL to the Rust bus engine, and provides cross-estimator comparison notebooks.

Minimal Proof
-------------

The equivalence script builds a minimal two-state, two-action MDP (Good/Bad states, Work/Relax actions) with discount factor 0.95 and shows that NFXP and MaxEnt IRL recover identical parameters from 1000 expert trajectories (200,000 state-action pairs).

Both methods converge to the same parameter vector and log-likelihood. The policy difference is zero to machine precision.

.. code-block:: text

   Ground Truth:  theta = [0.1000, 0.2000, 0.3000]
   NFXP (DDC):    theta = [0.0638, 0.1442, 0.3414]
   MaxEnt IRL:    theta = [0.0638, 0.1442, 0.3414]

   MSE(NFXP vs MaxEnt):  0.00e+00
   Log-likelihood:       -137444.51
   Max policy difference: 0.00e+00

The reward parameterization anchors one state-action pair (r(Bad, Relax) = 0) for identification following Kim et al. (2021). The remaining three parameters are free. Both methods find the same MLE because both optimize the conditional log-likelihood of observed actions given states under a softmax policy.

The full script is at ``examples/ddc-irl-equivalence/ddc_maxent_equivalence.py``.

Rust Bus Application
--------------------

MCE IRL (Ziebart 2010) applied to the original Rust bus engine data (9,410 observations, 90 buses). The demo script uses a two-feature specification (linear and quadratic mileage cost) and recovers the reward function with standard errors via numerical Hessian.

.. code-block:: text

   Parameter Estimates:
   Parameter                Estimate      Std Err     t-stat               95% CI
   linear_cost               11.0533       0.2056      53.77   [10.6504, 11.4562]
   quadratic_cost           -27.8992       0.8314     -33.56 [-29.5288, -26.2696]

   Log-Likelihood:           -3,216.98

The recovered reward function shows maintenance cost rising with mileage, peaking around 100,000 miles, then declining sharply as the engine ages. The replacement probability reaches near certainty above 250,000 miles.

The single-feature version (``mce_irl_bus_example.py``) uses only a linear mileage cost and confirms convergence with a different feature specification.

Scripts:

- ``examples/mce-irl/mce_irl_bus_demo.py`` -- two-feature MCE IRL with full output
- ``examples/mce-irl/mce_irl_bus_example.py`` -- single-feature MCE IRL with predictions
- ``examples/mce-irl/mce_irl_bus_example.ipynb`` -- interactive notebook version

Max-Margin Extension
--------------------

Max Margin IRL (Ratliff et al. 2006) maximizes the margin between expert features and the best alternative policy rather than maximizing likelihood. On the Rust bus problem, this produces parameters with incorrect signs because the margin objective does not penalize sign flips that increase the feature margin.

.. code-block:: text

   Max Margin IRL (Unit Norm):
     operating_cost:   -0.9999 (true: 0.0003)
     replacement_cost:  0.0026 (true: 1.0000)
     Cosine similarity: 0.0023

   MCE IRL (Likelihood-based):
     operating_cost:    0.001128 (true: 0.001000)
     replacement_cost:  2.999982 (true: 3.000000)

Max Margin IRL is better suited for imitation learning (policy matching) and feature selection than for structural parameter recovery. For parameter estimation with valid standard errors, MCE IRL is the correct choice.

The full script is at ``examples/mce-irl/max_margin_bus_example.py``.

Cross-Estimator Notebooks
-------------------------

These notebooks extend the comparison to Guided Cost Learning and provide a side-by-side NFXP versus MCE IRL benchmark on larger environments.

- ``examples/ddc-irl-equivalence/gcl_comparison.ipynb`` -- compares Guided Cost Learning against structural estimators
- ``examples/ddc-irl-equivalence/nfxp_vs_mceirl_comparison.ipynb`` -- side-by-side benchmark of NFXP and MCE IRL on shared environments
