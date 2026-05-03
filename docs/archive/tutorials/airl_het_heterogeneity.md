# AIRL-Het Anchor Identification and Heterogeneous Treatment Effects

| | |
|---|---|
| **Paper** | Lee, Sudhir, and Wang (2026). AIRL with Unobserved Heterogeneity for Structural DDC on Sequential Content |
| **Estimators** | AIRL-Het (K=2 segments, EM) vs NFXP (homogeneous) and AIRL with anchors (homogeneous) |
| **Environment** | Synthetic serialized content, 21 states, 3 actions (buy/wait/exit), 2 latent consumer segments |
| **Key finding** | Pooled estimators converge to a compromise reward that describes neither segment. AIRL-Het discovers the latent segments and enables heterogeneous counterfactual analysis. |

## Background

Standard structural estimators assume all individuals share the same utility function. When the population is a mixture of types with genuinely different preferences, pooled estimation recovers an average that describes nobody. A price increase that barely registers for one consumer type may drive another type to exit entirely. Without knowing the types, the pooled model predicts a moderate average effect, missing the divergent responses.

Lee, Sudhir, and Wang introduce the first heterogeneous adversarial IRL method. The estimator uses an EM algorithm where the E-step computes posterior segment probabilities from trajectory likelihoods under each segment's policy, and the M-step runs weighted AIRL for each segment with anchor constraints enforced. The anchors (exit action has reward zero, absorbing state has value zero) uniquely pin down the recovered rewards as true structural rewards rather than shaped equivalents, which is essential for counterfactual welfare analysis.

The paper validates its predictions against field experiment outcomes on a content platform with 500,000 users and discovers four latent consumer segments. The key practical finding is that heterogeneous treatment effects often point in opposite directions: a paywall policy increases purchases among quality-sensitive consumers by removing weak episodes from the free path, while simultaneously driving price-sensitive consumers to exit. The aggregate effect masks this divergence.

## Synthetic DGP

The tutorial constructs a two-segment DGP using the identification experiment infrastructure from ``experiments/identification/``. The serialized content environment has 20 regular episode states plus one absorbing state. Three actions are available at each state: buy (advance to the next episode), wait (stay at the current episode), and exit (enter the absorbing state).

Segment A ("Pay and Read") represents quality-sensitive consumers who buy frequently. They have a low barrier to purchasing (alpha_buy close to zero) and high sensitivity to content quality (theta_e large). Segment B ("Wait and Read") represents price-sensitive consumers who wait patiently. They have a high barrier to purchasing (alpha_buy strongly negative) and low sensitivity to content quality.

The mixing proportions are 40 percent Segment A and 60 percent Segment B. Each individual is drawn from one segment, and their trajectory is simulated under that segment's optimal policy. The pooled panel has no segment labels.

## Estimation

The companion script is at ``examples/serialized-content/airl_het_showcase.py``.

```python
from econirl.estimation.adversarial.airl_het import AIRLHetEstimator, AIRLHetConfig

airl_het = AIRLHetEstimator(config=AIRLHetConfig(
    num_segments=2,
    exit_action=2,
    absorbing_state=20,
    max_em_iterations=30,
    max_airl_rounds=100,
))
result = airl_het.estimate(panel, utility, problem, transitions)
```

The anchor constraints (exit action r=0, absorbing state V=0) are enforced automatically by the estimator when ``exit_action`` and ``absorbing_state`` are specified.

## Anchor Identification

The anchor identification theorem states that when an exit action with zero reward and an absorbing state with zero value exist in the data, the AIRL discriminator recovers the unique structural reward function rather than a shaped equivalent. Without anchors, AIRL only identifies rewards up to potential-based shaping h(s) minus beta times the expected h(s') under transitions, which preserves the optimal policy but changes the reward level. Counterfactual welfare calculations require the level, not just the ranking, so anchors are essential.

## Heterogeneous Counterfactuals

The tutorial demonstrates a Type C counterfactual: removing the wait-for-free option for the first 5 episodes (making them paid-only). The pooled model predicts a single moderate effect. AIRL-Het reveals that Pay-and-Read consumers are barely affected because they were already buying, while Wait-and-Read consumers see a spike in exit probability because their primary consumption strategy (waiting) is no longer available.

This divergence produces opposite policy recommendations depending on which segment the platform prioritizes. A platform optimizing total purchases sees a modest lift. A platform optimizing retention sees a large loss. The pooled model cannot distinguish these outcomes.

## When to Use AIRL-Het

AIRL-Het is designed for settings where the researcher suspects preference heterogeneity and needs segment-specific counterfactuals. The EM algorithm discovers the number and composition of segments from the data without requiring pre-labeled types. The anchor constraints guarantee that the recovered segment-specific rewards are structural (not shaped), enabling valid welfare comparisons across segments and across counterfactual policies.

The main computational cost is running K separate AIRL inner loops per EM iteration. On small tabular environments (20 to 100 states) this is fast. On large continuous-state problems, the adversarial training inner loop may need careful tuning.

AIRL-Het is the only estimator in econirl that combines reward recovery, preference heterogeneity, and anchor identification in a single estimation step. No other open-source library implements heterogeneous adversarial IRL.
