## Identification of Counterfactuals in Dynamic Discrete Choice Models (Kalouptsidi, Scott, and Souza-Rodrigues, 2021)

This note summarizes the paper and connects it to the artifacts in this repository.

Key ideas

- In dynamic discrete choice (DDC) models with additive T1EV shocks, observed conditional choice probabilities (CCPs) identify policy-relevant within-state action differences, but not reward levels. Rewards are set-identified up to a state-dependent additive function that cancels inside the softmax.
- Counterfactuals that change the environment (e.g., transitions or the action set) typically depend on the separation between immediate rewards and continuation values. CCPs alone are therefore insufficient to identify such counterfactuals.
- The paper formalizes conditions under which counterfactuals are identifiable and introduces the role of “additional restrictions,” such as:
  - Normalizations/anchors that pin the value-function offset (e.g., fixing the exit payoff and the absorbing-state value),
  - Exclusion restrictions that shift immediate utility but not transitions (or vice versa),
  - External variation (e.g., different transition laws) that allows one to disentangle flow utility from continuation values.

Connection here

- The AIRL-with-anchors (LSW) setup and the IQ-Learn/GLADIUS approaches implemented in this repo can be interpreted as imposing such additional restrictions to recover rewards that transfer across dynamics.
- Reduced-form logit over a flexible Q recovers the advantage function, not structural rewards; it therefore fails on Type B counterfactuals that require re-solving the Bellman equation under changed transitions.

Reference

- Kalouptsidi, M., Scott, P. T., and Souza-Rodrigues, E. (2021). Identification of Counterfactuals in Dynamic Discrete Choice Models. Quantitative Economics, 12(2), 351–403.

