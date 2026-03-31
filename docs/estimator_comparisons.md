# Estimator Comparisons & Dataset Mapping

`econirl` bridges dynamic discrete choice (DDC) models from structural econometrics with inverse reinforcement learning (IRL) from machine learning. This document outlines the 9 core estimation methods, their pros and cons, optimal use cases, and the specific datasets they are designed to analyze.

---

## 1. Structural MLE Estimators (Parametric Forward Models)

These classical estimators recover strictly specified structural utilities. They rely on explicit transition matrices and solving (or avoiding) the Bellman fixed-point.

### NFXP-NK (Nested Fixed Point with Newton-Kantorovich)
* **Core Mechanism**: Maximizes exact log-likelihood using a carefully optimized Bellman inner loop. Uses the SA→NK polyalgorithm and implicit differentiation for analytical gradients.
* **Pros**: Delivers statistically efficient Maximum Likelihood Estimation (MLE) with precise analytical standard errors. Guarantees a globally convergent, machine-precision solution.
* **Cons**: Computationally intense. Requires solving the Bellman equation $O(|S|^2)$ per outer step. Intractable for state spaces $>10,000$ or highly forward-looking settings ($\beta > 0.995$).
* **Where it shines**: Publication-grade structural estimates where confidence intervals and likelihood ratio tests are prioritized. 
* **Target Datasets**: Small tabular datasets. 
  * *Repo Datasets*: `rust_bus` (Harold Rust's canonical engine replacement data), `taxi-gridworld` (small discrete environments).

### CCP (Conditional Choice Probabilities / NPL)
* **Core Mechanism**: Hotz-Miller inversion. Uses empirical choice probabilities to bypass the Bellman inner loop.
* **Pros**: Rapid structural estimation via a single matrix inversion $O(|S|^3)$. Allows for specification searching without hours of compute, and handles dynamic games where Nash equilibrium computation is otherwise intractable.
* **Cons**: Still requires a full transition matrix and bounded state spaces. $K=1$ estimates are statistically inefficient.
* **Where it shines**: Rapid hypothesis testing, initial model selection, and dynamic oligopoly/game data.
* **Target Datasets**: Medium-sized tabular state spaces or large aggregates of finite choices.
  * *Repo Datasets*: `rust_bus`, classical multi-agent sequential choices.

---

## 2. Scalable Approximations (Continuous & Large State Spaces)

When exact tabular representation fails, these estimators integrate neural and basis-function approximations to manage state-space explosion.

### NNES (Neural Network Estimation of Structural Models)
* **Core Mechanism**: Optimizes the utility parameters alongside a Neural V-Network representing continuation values, relying on Neyman Orthogonality.
* **Pros**: Achieves continuous dataset scaling *with* valid standard errors. Orthogonality theoretically insulates structural parameters from neural network approximation noise. 
* **Cons**: Highly sensitive to initial parameter guessing. Achieving an exact zero Bellman residual in deep learning contexts is rare.
* **Where it shines**: High-dimensional, continuous state spaces prioritizing publication-grade inference over raw compute speed.
* **Target Datasets**: Complex panels with continuous state representations.
  * *Repo Datasets*: `ngsim` (continuous vehicle kinematics, velocity, headway tracking).

### TD-CCP (Temporal Difference CCP with Feature Decomposition)
* **Core Mechanism**: Modifies conditional choice probabilities with Neural Approximate Value Iteration (AVI). Separates the EV networks per structural utility feature.
* **Pros**: Highly interpretable diagnostics; by isolating networks for specific features (e.g., separating the "distance" value network from the "velocity" value network), researchers can debug exactly which component is failing to converge. Allows scaling to continuous/massive variables.
* **Cons**: Resource heavy—requires $K+1$ discrete networks. TD learning carries no strict global convergence guarantee here.
* **Where it shines**: Complex, multi-feature environments where continuous metrics govern behavior. 
* **Target Datasets**: Multi-dimensional trajectory data where tracking independent utility features is necessary.
  * *Repo Datasets*: `ngsim` (lane changing dynamics), `eth_ucy` / `stanford_drone` (pedestrian continuous trajectory features).

### SEES (Sieve-Based Structural Estimation)
* **Core Mechanism**: Bypasses neural nets entirely by projecting Value functions onto deterministic Sieve bases (Fourier/Chebyshev).
* **Pros**: $O(1)$ scaling with respect to the state-space size. Instantaneous structural approximation via a single L-BFGS-B call. 
* **Cons**: Vulnerable to projection bias; if value boundaries are mathematically sharp or discontinuous, the Sieve basis behaves poorly. Manual basis tuning is required.
* **Where it shines**: Giant panel datasets with completely smooth expected value curves where deep learning is overkill.
* **Target Datasets**: Extremely large categorical/continuous aggregate datasets.
  * *Repo Datasets*: Large-scale GPS aggregates like `tdrive`, `geolife`.

---

## 3. Inverse Reinforcement Learning (Demonstration-Driven)

These methods operate when the forward structure isn't easily defined, turning instead to expert demonstration trajectories to reverse-engineer matching rewards out of high-dimension real-world actions.

### MCE-IRL (Maximum Causal Entropy IRL) // MCE-IRL Deep
* **Core Mechanism**: Solves the soft-Bellman formulation to match analytical features inside demonstration data while enforcing predictive robustness (causal entropy).
* **Pros**: Bridging structural logit with machine learning. Returns interpretable linear reward weights, exact feature matching, and robust predictive behavior.
* **Cons**: Standard (Linear) MCE requires explicit transition matrices. The Deep subset trades parameter interpretability for handling nonlinear neural preferences.
* **Where it shines**: Recovering complex reward weights from massive expert demonstrations.
* **Target Datasets**: Trajectory logistics, routing, and transit data. 
  * *Repo Datasets*: `tdrive` (Beijing taxi logs), `geolife` (mobility traces), `citibike` (bike platform matching), `foursquare` (check-ins).

### AIRL (Adversarial Inverse Reinforcement Learning)
* **Core Mechanism**: Trains a discriminator to parse expert versus generated data while canceling out potential-based shaping networks, enforcing structural recovery instead of pure behavior cloning.
* **Pros**: Provides mathematical transfer guarantees. Recovered state rewards remain valid even when fundamental transition dynamics in downstream environments change. 
* **Cons**: Deeply unstable adversarial training loop. Solving the total MDP per iteration causes profound slowness.
* **Where it shines**: Sim-to-Real gaps. Learning on a clean platform variant but transferring the utility model to deployments with modified laws-of-physics/dynamics.
* **Target Datasets**: Autonomous navigation requiring safety guardrails across unseen intersections.
  * *Repo Datasets*: `ngsim`, `argoverse` (autonomous car routing variants), robotics datasets like `d4rl`.

### f-IRL (Feature-Free Distribution Matching)
* **Core Mechanism**: Bypasses defined features by using divergence metrics ($KL$, $TV$, $\chi^2$) to force probability matching of state-actions directly.
* **Pros**: Requires absolutely zero reward assumption. Eliminates feature engineering and discriminator instability. 
* **Cons**: Confined heavily to purely tabular mappings and requires robust density in expert observations across the full horizon. 
* **Where it shines**: Completely exploratory analysis. Exposes what states matter before any formal structural models are formed.
* **Target Datasets**: Small-scale exploration or grid mappings of new data.
  * *Repo Datasets*: `citibike` or `foursquare` station-action matrices (determining raw latent hotspots).

---

## 4. The Baseline

### BC (Behavioral Cloning)
* **Core Mechanism**: Straightforward supervised observation matching $P(a|s) = N(s,a)/|N(s)|$.
* **Pros**: Computationally instantaneous. Validates whether fundamental data signals even exist.
* **Cons**: No structural capability, zero transferability, and compounded $O(T^2\epsilon)$ error under distributional drift.
* **Where it shines**: The mandatory starting line. If structural estimators (MCE-IRL, NFXP) fail to beat BC on out-of-sample data, the environment fundamentally lacks sequential "forward-looking" structural intent. 
* **Target Datasets**: *All datasets.* Mandatory baseline test.