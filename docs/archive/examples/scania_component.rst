SCANIA Component X Replacement
===============================

.. image:: /_static/mdp_schematic_scania_component.png
   :alt: SCANIA component MDP structure showing degradation state chain with operate and replace actions.
   :width: 80%
   :align: center

The SCANIA IDA 2024 Industrial Challenge dataset tracks 23,550 heavy trucks with 105 anonymized operational readout features grouped under 14 sensor families. Each vehicle is observed at irregular intervals until either an anonymized mechanical component is repaired or the study window closes. Of the 23,550 vehicles, 2,272 (9.6 percent) received a repair during the study period. The remaining 21,278 are right-censored.

Model
-----

Let :math:`d_t \in \{0, 1, \ldots, S-1\}` denote the degradation state of the component at observation :math:`t`. At each observation the fleet manager chooses action :math:`a_t \in \{0, 1\}`, where :math:`a_t = 0` means keep and :math:`a_t = 1` means replace.

The per-period utility is

.. math::

   u(d_t, a_t; \theta) = \begin{cases} -\theta_c \, d_t + \varepsilon_t(0) & \text{if } a_t = 0 \\ -RC + \varepsilon_t(1) & \text{if } a_t = 1 \end{cases}

where :math:`\theta_c` is the operating cost per unit degradation, :math:`RC` is the fixed replacement cost, and :math:`\varepsilon_t(a)` are i.i.d. Type I Extreme Value preference shocks.

The manager maximizes the expected discounted sum of utilities

.. math::

   \max_{\{a_t\}} \; \mathbb{E} \left[ \sum_{t=0}^{\infty} \beta^t \, u(d_t, a_t; \theta) \right]

subject to transition probabilities :math:`P(d_{t+1} \mid d_t, a_t)`. If the manager keeps the component, degradation increases stochastically by 0, 1, or 2 bins. If the manager replaces, degradation resets to zero and then increases by 0, 1, or 2 bins. These transitions are estimated from the data.

The Bellman equation for the choice-specific value function is

.. math::

   v(d, a; \theta) = u(d, a; \theta) + \beta \sum_{d'} P(d' \mid d, a) \log \left( \sum_{a'} \exp\bigl(v(d', a'; \theta)\bigr) \right)

and the conditional choice probability under the logit assumption is

.. math::

   P(a \mid d; \theta) = \frac{\exp\bigl(v(d, a; \theta)\bigr)}{\sum_{a'} \exp\bigl(v(d, a'; \theta)\bigr)}

The structural parameters :math:`\theta = (\theta_c, RC)` are estimated by maximum likelihood. The log-likelihood is

.. math::

   \mathcal{L}(\theta) = \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P(a_{it} \mid d_{it}; \theta)

where :math:`N = 23{,}550` vehicles and :math:`T_i` is the number of observations for vehicle :math:`i`.

Differences from Rust (1987)
-----------------------------

This is a single-spell optimal stopping problem, not a renewal replacement problem. In Rust (1987), each bus is observed through multiple replacement cycles and continues accumulating mileage after each replacement. Here each vehicle contributes at most one repair event, and no post-replacement observations are recorded. The transition dynamics after replacement are therefore not identified from data and must be imposed by assumption.

The state space has 105 continuous sensor features rather than one scalar mileage reading. A principal component analysis on the standardized feature matrix shows that PC1 explains 97.0 percent of total variance and PC2 adds 2.5 percent. The 105 sensor readings are nearly collinear, so a single degradation axis captures almost all information. This PC1 score is discretized into 50 percentile-based bins to produce a finite state space for NFXP.

Observations arrive at irregular intervals rather than on a fixed monthly grid. The time steps are continuous floats ranging from 73 to 510, with between 5 and 303 observations per vehicle (median 43). The discount factor is set to 0.99 rather than 0.9999 to account for the longer and irregular spacing between decision periods.

.. image:: /_static/scania_component_overview.png
   :alt: SCANIA Component X data showing degradation distribution by action, empirical replacement rate by degradation bin, and time-to-event distribution by repair outcome.
   :width: 100%

Estimation
----------

.. code-block:: python

   from econirl import NFXP, NNES
   from econirl.datasets import load_scania

   df = load_scania(data_dir="data/scania/Dataset/")

   nfxp = NFXP(n_states=50, discount=0.99).fit(
       df, state="degradation_bin", action="replaced", id="vehicle_id"
   )
   nnes = NNES(n_states=50, discount=0.99, bellman="npl").fit(
       df, state="degradation_bin", action="replaced", id="vehicle_id"
   )

.. list-table::
   :header-rows: 1

   * - Estimator
     - :math:`\hat{\theta}_c`
     - :math:`\widehat{RC}`
     - :math:`\text{SE}(\hat{\theta}_c)`
     - :math:`\text{SE}(\widehat{RC})`
     - LL
     - Time
   * - NFXP (23,550 vehicles)
     - 0.0016
     - 8.5141
     - 0.0001
     - 0.1050
     - -15,745
     - 148s
   * - NNES (2,000 vehicles)
     - 0.0533
     - 8.1754
     - 0.0006
     - 0.0005
     - -1,429
     - 326s

Results
-------

Both parameters are statistically significant at the 0.1 percent level under NFXP. The operating cost is 0.0016 per degradation bin per period, so a truck at bin 40 pays :math:`0.0016 \times 40 = 0.064` in operating costs each period. The replacement cost is 8.51. Replacement becomes attractive when the expected present value of future operating costs at the current degradation level exceeds this threshold. The replacement cost is roughly three times the Rust (1987) estimate of 3.0, which is consistent with the higher cost of heavy truck component repairs relative to bus engine overhauls.

The mean degradation bin at replacement events is 34.3, compared to 24.5 at keep decisions. This 9.9-bin gap is consistent with a threshold replacement policy where the manager waits until degradation is sufficiently high before paying the fixed replacement cost.

NFXP and NNES agree on the replacement cost (8.51 versus 8.18) but disagree on the operating cost (0.0016 versus 0.053). NFXP computes the exact value function by solving the Bellman fixed point on the 50-state transition matrix. NNES approximates the value function with a neural network trained via the NPL Bellman with Hotz-Miller inversion. The neural approximation absorbs part of the state-dependent operating cost gradient into the flexible V-network, which inflates the :math:`\hat{\theta}_c` estimate. On a 50-state problem where exact solution is feasible, NFXP is the more reliable estimator.

The Hessian condition number for NFXP is :math:`4.9 \times 10^7`, indicating that the likelihood surface is elongated along the direction that trades off :math:`\theta_c` against :math:`RC`. This is typical of replacement models where the two cost parameters are only separately identified through the curvature of the replacement probability as a function of degradation.

Counterfactual analysis
-----------------------

Once the structural parameters are estimated, the model can answer counterfactual questions. What would the fleet manager do if the replacement cost were halved or doubled? The table below shows the replacement probability at selected degradation bins under the baseline estimate (:math:`\widehat{RC} = 8.31`) and two counterfactual scenarios, estimated on a 2,000-vehicle subset.

.. code-block:: python

   cf_half = nfxp.counterfactual(RC=rc_est / 2)
   cf_double = nfxp.counterfactual(RC=rc_est * 2)

.. list-table::
   :header-rows: 1

   * - Degradation bin
     - Baseline
     - RC / 2
     - RC * 2
   * - 0
     - 0.0002
     - 0.0154
     - 0.0000
   * - 10
     - 0.0006
     - 0.0233
     - 0.0000
   * - 20
     - 0.0014
     - 0.0325
     - 0.0000
   * - 30
     - 0.0027
     - 0.0426
     - 0.0000
   * - 40
     - 0.0040
     - 0.0527
     - 0.0000
   * - 49
     - 0.0047
     - 0.0583
     - 0.0000

Halving the replacement cost increases the replacement probability at every degradation level by an order of magnitude. At degradation bin 40, the probability rises from 0.40 percent to 5.27 percent. The manager replaces more frequently because the cost of resetting the component is lower, which dominates the forward-looking calculation. Doubling the replacement cost suppresses replacement almost entirely. No vehicles are replaced because the one-time cost exceeds the expected present value of future operating savings from a fresh component at any observed degradation level.

The welfare elasticity of replacement cost is negative 5.3. A 10 percent increase in replacement cost reduces expected lifetime utility by roughly 0.34 utils on average, with most of the welfare loss concentrated at high degradation states where the manager would otherwise have replaced.

GLADIUS
-------

GLADIUS is a model-free estimator that learns Q-values and expected continuation values via neural networks trained on observed transitions. It does not require a transition matrix. After training, the structural parameters are recovered by projecting the implied neural rewards :math:`\hat{r}(s,a) = \hat{Q}(s,a) - \beta \hat{E}V(s,a)` onto the linear feature specification.

Two modifications are needed for the SCANIA data. First, the per-period replacement rate is 0.20 percent, so the NLL loss is dominated by keep events. Inverse-frequency class weighting (459x weight on replace events) ensures the Q-network learns from the rare replacement signal. Second, the absolute level of Q-values is not identified in the model-free setting, so a constant reward like RC gets absorbed into the Q/EV decomposition. Projecting reward differences :math:`\hat{r}(s,1) - \hat{r}(s,0)` onto feature differences :math:`\varphi(s,1) - \varphi(s,0)` eliminates the unidentified constant and recovers both parameters. This is the neural analog of the reward normalization in Kim et al. (2021).

.. list-table::
   :header-rows: 1

   * - Estimator
     - :math:`\hat{\theta}_c`
     - :math:`\widehat{RC}`
     - R-squared
   * - NFXP
     - 0.0016
     - 8.51
     -
   * - GLADIUS
     - 0.069
     - 5.24
     - 0.92

GLADIUS recovers a positive operating cost and a positive replacement cost, both significant at the 0.1 percent level. The replacement cost is lower than the NFXP estimate (5.24 versus 8.51) and the operating cost is higher (0.069 versus 0.0016). This tradeoff reflects the difference between structural MLE (which uses the full Bellman fixed point) and the model-free projection (which relies on neural reward approximation). The R-squared of 0.92 indicates that the linear reward specification captures most of the variation in the neural implied rewards, though some nonlinearity remains.

Running the example
-------------------

.. code-block:: bash

   kaggle datasets download -d tapanbatla/scania-component-x-dataset-2025 -p data/scania --unzip
   python examples/scania-component/scania_nfxp.py --data-dir data/scania/Dataset/

Reference
---------

SCANIA Component X dataset, IDA 2024 Industrial Challenge. Kaggle: tapanbatla/scania-component-x-dataset-2025.

Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. Econometrica, 55(5), 999-1033.
