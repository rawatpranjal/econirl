# Appendix A - RLHF Papers Collection

This directory contains key papers for understanding the theoretical foundations of Appendix A, which establishes the equivalence between NFXP (DDC), MaxEnt-IRL, and RLHF under the soft-control framework.

## Core RLHF Papers

### 1. Ouyang et al. (2022) - InstructGPT/ChatGPT
**File**: `ouyang2022_instructgpt.pdf` (1.7M)
**Title**: Training language models to follow instructions with human feedback
**arXiv**: 2203.02155
**Description**: The seminal paper introducing InstructGPT and the RLHF methodology used in ChatGPT. Uses Bradley-Terry preference model with PPO optimization.

### 2. Christiano et al. (2017) - Original RLHF
**File**: `christiano2017_rlhf_robotics.pdf` (3.1M)
**Title**: Deep reinforcement learning from human preferences
**arXiv**: 1706.03741
**Description**: The original RLHF paper applied to robotics tasks. Introduced learning reward functions from pairwise preference comparisons.

### 3. Schulman et al. (2017) - PPO Algorithm
**File**: `schulman2017_ppo.pdf` (2.8M)
**Title**: Proximal Policy Optimization Algorithms
**arXiv**: 1707.06347
**Description**: The PPO algorithm used in RLHF implementations (Section 5.1, line 30). Optimizes policies through clipped policy updates approximating the KL-regularized objective.

## Survey Paper

### 4. Kaufmann et al. (2023) - RLHF Survey
**File**: `kaufmann2023_rlhf_survey.pdf` (1.3M)
**Title**: A Survey of Reinforcement Learning from Human Feedback
**arXiv**: 2312.14925
**Description**: Comprehensive survey of RLHF methods and applications (cited in RLHF.tex:4).

## Mathematical Foundation

### 5. Haarnoja et al. (2017) - Soft Q-Learning
**File**: `haarnoja2017_soft_q_learning.pdf` (3.5M)
**Title**: Reinforcement learning with deep energy-based policies
**arXiv**: 1702.08165
**Description**: Introduces soft Q-learning with entropy regularization. Foundation for the soft Bellman equations in Appendix A.2 (cited as `softqlearning:2017`).

## DDC ↔ IRL Bridge

### 6. Kang, Yoganarasimhan & Jain (2025) - DDC & IRL Connection
**File**: `kang2025_ddc_irl.pdf` (1.1M)
**Title**: An Empirical Risk Minimization Approach for Offline Inverse RL and Dynamic Discrete Choice Model
**arXiv**: 2502.14131
**Description**: **HIGHLY RELEVANT** - Directly addresses offline IRL and DDC model connections, bridging econometrics and machine learning perspectives (cited in Introduction.tex:103).

---

## How These Papers Connect to Appendix A

**Appendix A** establishes that NFXP (DDC), MaxEnt-IRL, and RLHF are mathematically equivalent under the soft-control framework:

- **Soft-Control Framework (A.2)**: Based on entropy-regularized MDPs → Haarnoja (2017)
- **DDC Micro-foundation (A.3)**: EV1 shocks yield soft Bellman equations
- **Three Methods (A.4)**:
  - NFXP: Maximum likelihood from choice data
  - MaxEnt-IRL: Matching feature expectations
  - RLHF: Bradley-Terry preferences → Christiano (2017), Ouyang (2022)
- **Theorem (A.6)**: All three recover the same policy under appropriate gauges
- **Implementation**: PPO is used in practice → Schulman (2017)

## Additional Papers in Directory

You may also find these papers from earlier downloads:
- `Haarnoja_2017_ICML_Proceedings.pdf` - Alternative version
- `Haarnoja_2017_Soft_Q_Learning.pdf` - Duplicate
- `Ng_Harada_Russell_1999_Reward_Shaping.pdf` - Reward shaping (Appendix A.5)
- `Ng_Russell_2000_IRL.pdf` - Original IRL algorithms (Appendix A.5)
- `Rust_1987_Bus_Engines.pdf` - NFXP algorithm (Appendix A.4)

---

**Downloaded**: 2025-10-15
**Purpose**: Reference materials for Appendix A theoretical foundations
