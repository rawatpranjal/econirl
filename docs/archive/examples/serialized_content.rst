Serialized Content Consumption
===============================

This example replicates the key results of Lee, Sudhir, and Wang (2026) on a semi-synthetic serialized content environment. Users on a wait-for-free fiction platform decide at each episode whether to pay for immediate access, wait for free access after a delay, or exit the series. Two latent consumer segments have different reward parameters. AIRL-Anchored recovers the segments and rewards from pooled data without observing segment labels, then three counterfactual exercises demonstrate the value of heterogeneous estimation for platform policy design.

The state space has 51 states representing 50 episodes across 5 books (10 episodes each) plus one absorbing state. Episodes have synthetic quality and engagement scores that vary across the narrative arc. Quality peaks near the end of each book to create cliffhanger dynamics. The three actions are buy (pay and advance), wait (access free next period), and exit (enter absorbing state). Transitions are deterministic conditional on the action.

Identification via anchors
--------------------------

The standard IRL identification problem is that rewards are only identified up to an additive value function. With a single agent in a single environment, the observed policy is consistent with infinitely many reward functions. Lee et al. resolve this with anchor normalization rather than cross-environment variation.

The exit action serves as the identification anchor. The reward for exiting is set to zero at every state, and the value of the absorbing state is set to zero. These two normalizations pin down the structural reward uniquely for each latent segment, enabling welfare-valid counterfactual analysis. This is the "AIRL-Anchored" approach. It bypasses the need for the value-distinguishing condition of Cao et al. (2021), which would require verifying a rank condition on the transition matrices across all 5 books. The anchors work from a single environment and do not depend on the diversity of narrative structures across books.

Segments and ground truth
-------------------------

Two consumer segments generate the data. Segment A ("Pay and Read", 30 percent of users) has high quality sensitivity, strong cliffhanger response, low price sensitivity, and a completion drive that increases with book progress. Segment B ("Wait and Read", 70 percent of users) has high price sensitivity, weak cliffhanger response, low wait cost reflecting patience, and low completion drive.

Six features parameterize the reward function. The buy action carries buy cost, quality, cliffhanger, and book progress features. The wait action carries wait cost and quality features. The exit action carries no features, which enforces the anchor normalization.

Each user reads 3 different books, providing within-user consistency signal for the EM algorithm. The EM E-step assigns posterior segment probabilities per trajectory, then interpolates across trajectories of the same user to enforce consistent segment membership. The panel contains 2,000 users generating 4,010 trajectories with 26,820 total observations.

.. code-block:: python

   from econirl.estimation.adversarial.airl_het import AIRLHetEstimator, AIRLHetConfig

   config = AIRLHetConfig(
       num_segments=2,
       exit_action=2,           # anchor: r(s, exit) = 0 for all s
       absorbing_state=50,      # anchor: V(absorbing) = 0
       max_em_iterations=20,
       max_airl_rounds=100,
       reward_lr=0.05,
       consistency_weight=0.5,  # within-user segment consistency
   )
   het_result = AIRLHetEstimator(config).estimate(panel, utility, problem, transitions)

Counterfactual exercises
------------------------

Three counterfactual experiments modify the platform's pricing and access policy, re-solve the Bellman equation under the modified environment, and simulate behavior per segment. Following Lee et al. line 646, each counterfactual "re-optimizes the decision policy using the learned reward function within the modified MDP environment."

**Type A (Introduce WFF).** The baseline is a pay-only platform where the wait action is disabled (equivalent to exit). Introducing wait-for-free increases consumption for the Wait and Read segment by 90 percent because patient users who would have exited under pay-only now have a free path through the series. The Pay and Read segment shows a modest 3 percent increase because they already preferred buying.

**Type B (Segment-customized wait-times).** Increasing the wait cost for Pay and Read (longer delay, screens them into buying) while decreasing it for Wait and Read (shorter delay, sustains engagement) increases Wait and Read purchases by 8 percent while barely affecting Pay and Read purchases. A pooled model would predict a uniform effect, missing the fact that the two segments respond in opposite directions.

**Type C (Content-based pricing).** Cliffhanger episodes (22 out of 50) become paid-only with the wait option disabled, while weak episodes become free with zero buy cost. Wait and Read purchases increase by 356 percent because users who previously waited through the entire series are now forced to buy at the cliffhangers to continue reading. Pay and Read consumption increases by 2 percent because free weak episodes reduce churn through low-engagement stretches. This is the most powerful counterfactual because it exploits content variation that only a heterogeneous model with episode-level features can leverage.

.. list-table:: Counterfactual Results
   :header-rows: 1

   * - Counterfactual
     - Pay and Read
     - Wait and Read
   * - Type A: consumption change
     - +2.9%
     - +90.3%
   * - Type B: purchase change
     - -0.1%
     - +8.3%
   * - Type C: purchase change
     - +1.6%
     - +356.4%

The Type C result illustrates why heterogeneous estimation matters. A pooled model would predict a moderate increase in purchases for the average user. The segment-level analysis reveals that the effect is driven almost entirely by the Wait and Read segment, whose behavior changes dramatically under content-based pricing.

Field experiment validation
---------------------------

A simulated field experiment reduces the wait cost on held-out books 3 and 4 (episodes 30 through 49) by 50 percent. Under the reduced wait cost, Pay and Read users show almost no change in buying behavior but increase waits by 0.77 per book. Wait and Read users increase waits by 0.21 and slightly increase purchases. This confirms that the recovered rewards predict directionally correct behavioral responses under policy changes not seen during estimation, matching the validation approach in Lee et al. Section 6.

Running the example
-------------------

.. code-block:: bash

   python examples/serialized-content/lsw_replication.py

Reference
---------

Lee, P., Sudhir, K., Wang, Y. (2026). Modeling Serialized Content Consumption: Adversarial IRL for Dynamic Discrete Choice.

Fu, J., Luo, K., Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. ICLR.

Cao, Y., Cohen, S.B. (2021). Identifiability in Inverse Reinforcement Learning. NeurIPS.
