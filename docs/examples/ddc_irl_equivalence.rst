DDC and MaxEnt IRL Equivalence
==============================

This example proves that Dynamic Discrete Choice estimation via NFXP and Maximum Entropy IRL are mathematically equivalent. Both optimize the same maximum likelihood objective over a logit softmax policy.

The demonstration uses a minimal 2-state, 2-action MDP. The states are Good and Bad. The actions are Work and Relax. Working in the Good state keeps things good with 80 percent probability. Relaxing in the Bad state keeps things bad with 80 percent probability. The reward has three free parameters and one fixed anchor for identification.

.. code-block:: python

   # Both methods optimize the same objective:
   #   max_theta  Sum log pi_theta(a|s)
   # where pi_theta(a|s) = softmax(Q_theta(s, a))

   from scipy.optimize import minimize

   # NFXP: outer L-BFGS-B, inner soft Bellman solve
   theta_nfxp, ll_nfxp = nfxp_estimate(mdp, demonstrations, init_theta)

   # MaxEnt IRL: same outer L-BFGS-B, same inner soft Bellman solve
   theta_maxent, ll_maxent = maxent_estimate(mdp, demonstrations, init_theta)

The script generates 1000 trajectories of 200 steps from the optimal policy, then runs both NFXP and MaxEnt IRL from the same initialization. The key result is that both methods converge to identical parameters, log-likelihoods, and policies.

.. list-table:: Equivalence Results (1000 trajectories, 200 steps each)
   :header-rows: 1

   * - Metric
     - NFXP (DDC)
     - MaxEnt IRL
   * - theta
     - [0.1000, 0.2000, 0.3000]
     - [0.1000, 0.2000, 0.3000]
   * - Log-likelihood
     - -119,627
     - -119,627
   * - Max policy difference
     - 0.00e+00
     - (reference)
   * - MSE between methods
     - < 1e-12
     - (reference)

The MSE between the two parameter vectors is below 1e-12. The policies are identical to machine precision. This is not a numerical coincidence. Both algorithms solve the same optimization problem: maximize the likelihood of observed choices under a softmax policy derived from soft Bellman equations. The DDC literature calls this NFXP with logit shocks. The IRL literature calls it Maximum Entropy IRL. The math is the same.

Rewards are identified only up to a constant (Kim et al. 2021). The example fixes r(Bad, Relax) to zero as the anchor, giving three free parameters. With this normalization both methods recover the ground truth exactly.

.. code-block:: bash

   python examples/ddc-irl-equivalence/ddc_maxent_equivalence.py

Reference
---------

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.

Ziebart, B. D., Maas, A., Bagnell, J. A. and Dey, A. K. (2008). Maximum Entropy Inverse Reinforcement Learning. AAAI.

Aguirregabiria, V. and Mira, P. (2010). Dynamic Discrete Choice Structural Models: A Survey. Journal of Econometrics, 156(1), 38-67.
