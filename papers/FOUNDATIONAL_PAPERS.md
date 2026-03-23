# Foundational Papers in IRL and DDC

This document indexes the foundational papers bridging Inverse Reinforcement Learning (IRL) and Dynamic Discrete Choice (DDC) methodologies.

## Paper Index

| File | Authors | Year | Key Contribution |
|------|---------|------|------------------|
| `2000_ng_russel_inverse_reinforcement_learning.pdf` | Ng & Russell | 2000 | Original IRL formulation via linear programming |
| `2004_abbeel_ng_apprenticeship_learning.pdf` | Abbeel & Ng | 2004 | Apprenticeship learning via feature matching |
| `2008_maximum_entropy_inverse_reinforcement_learning.pdf` | Ziebart et al. | 2008 | Maximum entropy IRL framework |
| `2016-generative-adversarial-imitation-learning-Paper.pdf` | Ho & Ermon | 2016 | GAIL: adversarial approach to imitation learning |
| `2018_adversarial_IRL.pdf` | Fu et al. | 2018 | AIRL: recoverable reward functions |
| `AguirregabiriaMira_ECMA2002.pdf` | Aguirregabiria & Mira | 2009 | NPL algorithm for dynamic discrete games* |
| `2010_Aguirregabiria_Mira_DDC_Survey.pdf` | Aguirregabiria & Mira | 2010 | Comprehensive DDC methodology survey |
| `2022_zeng_irl_and_ddc.pdf` | Zeng, Hong, Garcia | 2022 | Structural estimation of MDPs with IRL |
| `2024_zeng_et_al_reinvention_of_NFXP_from_IRL_perspective.pdf` | Zeng et al. | 2023 | ML framework for offline IRL |
| `2023_geng_ddc_irl.pdf` | Geng et al. | 2023 | State aggregation for DDC via IRL |
| `2022_iq_learn.pdf` | Garg et al. | 2021 | IQ-Learn: implicit Q-learning for IRL |
| `2017-deep-reinforcement-learning-from-human-preferences-Paper.pdf` | Christiano et al. | 2017 | RLHF: learning from human preferences |
| `2021_identifiability_in_inverse_reinforcement_learning.pdf` | Cao et al. | 2021 | Identifiability conditions for IRL |

*Note: Original A&M 2002 Econometrica paper is paywalled. This file contains related NPL methodology working paper.

## Categories

### Core IRL Algorithms
Foundational IRL methods that established the field.

1. **Ng & Russell (2000)** - The original IRL paper. Formulates IRL as linear programming: find rewards that make observed behavior optimal.
2. **Abbeel & Ng (2004)** - Apprenticeship learning. Matches feature expectations between expert and learned policy.
3. **Ziebart et al. (2008)** - Maximum entropy IRL. Resolves ambiguity by preferring policies with maximum entropy, connecting to logit choice models.

### Adversarial/Generative Approaches
Methods using adversarial training for imitation and reward learning.

4. **Ho & Ermon (2016)** - GAIL. Casts imitation as occupancy measure matching via GAN-style training.
5. **Fu et al. (2018)** - AIRL. Extends GAIL to recover disentangled, transferable reward functions.

### DDC/Econometrics Foundations
Structural estimation methods from economics.

6. **Aguirregabiria & Mira (2009)** - NPL algorithm. Nested pseudo-likelihood for dynamic discrete games (extends 2002 CCP method).
7. **Aguirregabiria & Mira (2010)** - DDC survey. Comprehensive review of DDC estimation methods including NFXP, CCP, and finite dependence.

### Bridging DDC and IRL
Papers establishing formal connections between the two fields.

8. **Zeng, Hong, Garcia (2022)** - Structural estimation of MDPs. Proves equivalence between DDC and ML-IRL, proposes finite-time algorithms.
9. **Zeng et al. (2023)** - ML framework for offline IRL. Maximum likelihood approach with demonstrations.
10. **Geng et al. (2023)** - State aggregation. Data-driven method combining IRL with DDC estimation.

### Modern Extensions
Recent advances building on foundational work.

11. **Garg et al. (2021)** - IQ-Learn. Single Q-function approach avoiding explicit reward recovery.
12. **Christiano et al. (2017)** - RLHF. Learning rewards from human preference comparisons.
13. **Cao et al. (2021)** - Identifiability. Conditions under which reward functions can be uniquely recovered.

## Recommended Reading Order

### For IRL Background
1. Ng & Russell (2000) - Foundational problem setup
2. Abbeel & Ng (2004) - Practical algorithm
3. Ziebart et al. (2008) - Maximum entropy framework
4. Ho & Ermon (2016) - Modern adversarial approach
5. Fu et al. (2018) - Reward recovery

### For DDC Background
1. Aguirregabiria & Mira (2010) - Comprehensive survey (read this first)
2. Aguirregabiria & Mira (2009) - NPL algorithm details

### For DDC-IRL Connection
1. Zeng, Hong, Garcia (2022) - Structural estimation bridge
2. Geng et al. (2023) - State aggregation approach
3. Zeng et al. (2023) - ML-IRL framework

### For Modern Methods
1. Garg et al. (2021) - IQ-Learn
2. Christiano et al. (2017) - RLHF
3. Cao et al. (2021) - Identifiability theory

## Key Insights

**IRL and DDC Equivalence**: Both fields solve the inverse problem of inferring preferences from observed behavior. DDC assumes parametric utility functions and uses MLE; IRL uses reward functions with various regularization schemes.

**Maximum Entropy Connection**: Ziebart's MaxEnt IRL is equivalent to logit-based DDC models. The entropy regularization in IRL corresponds to the Type-I extreme value errors in discrete choice.

**CCP and Policy**: Conditional choice probabilities in DDC are analogous to policies in RL. Both represent the probability of taking actions given states.
