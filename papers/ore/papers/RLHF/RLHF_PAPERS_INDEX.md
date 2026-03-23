# RLHF Papers Collection
**Reinforcement Learning from Human Feedback - Literature Survey**

Organization of recent RLHF papers to complement Section 5 (RLHF) of the Oxford Research Encyclopedia article on "Structural Econometrics and Inverse Reinforcement Learning."

---

## 📚 Overview

This collection focuses on **new perspectives** beyond what our paper already covers:
- Our paper: Theoretical equivalence between per-step RLHF ↔ DDC ↔ MaxEnt-IRL
- This collection: Practical advances, alternatives, safety, efficiency, personalization

---

## 1. Foundational Papers (`foundational/`)

### Christiano et al. (2017)
**Title**: Deep Reinforcement Learning from Human Preferences
**URL**: https://arxiv.org/abs/1706.03741
**Key Contribution**: Demonstrated that complex behaviors (Atari, robotics) can be learned from binary preference feedback
**Relevance**: Cited in our RLHF section as foundational work

### Ouyang et al. (2022) - InstructGPT
**Title**: Training language models to follow instructions with human feedback
**URL**: https://arxiv.org/abs/2203.02155
**Key Contribution**: 3-stage RLHF pipeline (SFT → reward model → PPO), precursor to ChatGPT
**Relevance**: Core citation in our Section 5, demonstrates sequence-level RLHF in practice

### Askell et al. (2021)
**Title**: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
**URL**: https://arxiv.org/abs/2204.05862
**Key Contribution**: Anthropic's work balancing helpfulness vs harmlessness objectives
**Relevance**: Explores multi-objective RLHF, beyond single reward model

---

## 2. Surveys & Critical Analysis (`surveys/`)

### Kaufmann et al. (2023) ⭐
**Title**: A Survey of Reinforcement Learning from Human Feedback
**URL**: https://arxiv.org/abs/2312.14925
**ArXiv PDF**: https://arxiv.org/pdf/2312.14925.pdf
**Key Contribution**: Comprehensive overview of RLHF landscape, PbRL origins, diverse applications
**Relevance**: **Already cited in our RLHF section** (footnote after line 4)
**Status**: Should download for reference

### Abdelnabi et al. (2024)
**Title**: RLHF Deciphered: A Critical Analysis of Reinforcement Learning from Human Feedback for LLMs
**URL**: https://arxiv.org/abs/2404.08555
**ArXiv HTML**: https://arxiv.org/html/2404.08555
**Key Contribution**: Critical examination of RLHF assumptions, reward model limitations
**Relevance**: Complements our theoretical treatment with practical concerns

### DPO Survey (2024)
**Title**: A Comprehensive Survey of Direct Preference Optimization: Datasets, Theories, Variants, and Applications
**Semantic Scholar**: https://www.semanticscholar.org/paper/ca47e76a16ec212674c6c57db37735b680a853e8
**Key Contribution**: Reviews DPO as RL-free alternative to PPO stage
**Relevance**: **Major gap in our paper** - DPO bypasses per-step RL entirely

---

## 3. Alternatives to RL (`alternatives_to_RL/`)

### Direct Preference Optimization (DPO)
**Why Important**: Our Section 5.6 mentions PPO as standard implementation, but DPO has emerged as simpler alternative

#### Exploratory Preference Optimization (XPO) (2024)
**Title**: Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF
**URL**: https://arxiv.org/abs/2405.21046
**ArXiv PDF**: http://arxiv.org/pdf/2405.21046.pdf
**Key Contribution**: Online DPO variant with provable exploration guarantees
**Relevance**: Extends DPO to sample-efficient online setting

---

## 4. Reward Modeling Advances (`reward_modeling/`)

### Token-Level Continuous Reward (TLCR) (2024)
**Title**: TLCR: Token-Level Continuous Reward for Fine-grained Reinforcement Learning from Human Feedback
**URL**: https://arxiv.org/abs/2407.16574
**Key Contribution**: Continuous rewards per token (not discrete +/-/0), addresses sequence vs per-step mismatch
**Relevance**: **Directly relevant to our Section 5.4** - refines per-step reward modeling

### R3HF (2024)
**Title**: R3HF: Reward Redistribution for Enhancing Reinforcement Learning from Human Feedback
**URL**: https://arxiv.org/html/2411.08302
**ArXiv HTML**: https://arxiv.org/html/2411.08302
**Key Contribution**: Reward redistribution technique for better alignment
**Relevance**: Improves reward model quality

### Dense Reward for Free (2024)
**Title**: Dense Reward for Free in RLHF
**URL**: https://arxiv.org/abs/2402.09401
**Key Contribution**: Uses attention weights to create dense rewards without extra labeling cost
**Relevance**: Practical efficiency improvement

---

## 5. Safety & Risk-Awareness (`safety_risk/`)

### RA-PbRL (2025) ⭐
**Title**: RA-PbRL: Provably Efficient Risk-Aware Preference-Based Reinforcement Learning
**URL**: https://arxiv.org/abs/2410.23569
**ArXiv HTML**: https://arxiv.org/html/2410.23569v2
**Key Contribution**: Risk-aware RLHF (CVaR optimization), theoretical guarantees
**Relevance**: **Extends our equivalence theorem** - risk-aware objectives beyond mean reward

### Safe RLHF (2023)
**Title**: Safe RLHF: Safe Reinforcement Learning from Human Feedback
**URL**: https://arxiv.org/abs/2310.12773
**ArXiv PDF**: https://arxiv.org/pdf/2310.12773.pdf
**Key Contribution**: Separates helpfulness reward model from harmlessness cost model
**Relevance**: Multi-objective RLHF with safety constraints

### Robust RLHF (2024)
**Title**: Robust Reinforcement Learning from Corrupted Human Feedback
**URL**: https://arxiv.org/abs/2406.15568
**Key Contribution**: Robustness to noisy/incorrect preference labels
**Relevance**: Addresses identification issues mentioned in our Section 5.2

### Zero-Shot LLMs in Human-in-the-Loop RL (2025)
**Title**: Zero-Shot LLMs in Human-in-the-Loop RL: Replacing Human Feedback for Reward Shaping
**URL**: https://arxiv.org/abs/2503.22723
**ArXiv HTML**: https://arxiv.org/html/2503.22723v1
**Key Contribution**: LLMs flag and correct biases in human feedback
**Relevance**: Meta-level approach to improve feedback quality

---

## 6. Efficiency & Scalability (`efficiency_scalability/`)

### Asynchronous RLHF (2025)
**Title**: Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models
**URL**: https://arxiv.org/abs/2410.18252
**ArXiv PDF**: https://arxiv.org/pdf/2410.18252.pdf
**Key Contribution**: Off-policy asynchronous training (vs on-policy PPO)
**Relevance**: Alternative to PPO mentioned in our Section 5.6

### RLAIF (2024) ⭐
**Title**: RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
**URL**: https://arxiv.org/abs/2309.00267
**ArXiv PDF**: http://arxiv.org/pdf/2309.00267.pdf
**Key Contribution**: AI-generated preferences match human feedback performance
**Relevance**: Addresses Bradley-Terry model data collection (Section 5.2)

### Active Queries in RLHF (2024)
**Title**: Reinforcement Learning from Human Feedback with Active Queries
**URL**: https://arxiv.org/abs/2402.00782
**Key Contribution**: Active learning to select most informative preference queries
**Relevance**: Efficient preference data collection

### Uni-RLHF (2024)
**Title**: Uni-RLHF: Universal Platform and Benchmark Suite for Reinforcement Learning with Diverse Human Feedback
**URL**: https://arxiv.org/abs/2402.02423
**ArXiv PDF**: https://arxiv.org/pdf/2402.02423.pdf
**Key Contribution**: Open-source platform and benchmarks for RLHF methods
**Relevance**: Standardization and evaluation infrastructure

---

## 7. Personalization (`personalization/`)

### Poddar et al. (2024)
**Title**: Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning
**URL**: https://arxiv.org/abs/2408.10075
**Key Contribution**: User-specific latent variables for heterogeneous preferences
**Relevance**: Extends single-reward-model assumption in our Section 5.2

### P-ShareLoRA (2025)
**Title**: A Shared Low-Rank Adaptation Approach to Personalized RLHF
**URL**: https://arxiv.org/abs/2503.19201
**ArXiv PDF**: https://arxiv.org/pdf/2503.19201.pdf
**Key Contribution**: LoRA for efficient personalized reward models
**Relevance**: Practical implementation of personalization

### Curiosity-Driven RLHF (2025)
**Title**: Curiosity-Driven Reinforcement Learning from Human Feedback
**URL**: https://arxiv.org/abs/2501.11463
**ArXiv HTML**: https://arxiv.org/html/2501.11463v1
**Key Contribution**: Combines RLHF with exploration to maintain output diversity
**Relevance**: Addresses potential mode collapse in RLHF

---

## 8. Multimodal Extensions (`multimodal/`)

### MM-RLHF (2025)
**Title**: MM-RLHF: The Next Step Forward in Multimodal LLM Alignment
**URL**: https://arxiv.org/abs/2502.10391
**Key Contribution**: 120K human preference pairs for multimodal models (text + images)
**Relevance**: Extends RLHF beyond text generation (Section 5 focus)

---

## 9. Federated & Privacy-Preserving (`federated_privacy/`)

### FedRLHF (2025)
**Title**: FedRLHF: A Convergence-Guaranteed Federated Framework for Privacy-Preserving and Personalized RLHF
**URL**: https://arxiv.org/abs/2412.15538
**ArXiv PDF**: http://arxiv.org/pdf/2412.15538.pdf
**Key Contribution**: Decentralized RLHF with convergence guarantees
**Relevance**: Privacy-preserving alternative to centralized reward modeling

---

## 📊 Connection to Our Paper

### What We Cover (Section 5):
- ✅ Sequence-level vs per-step RLHF formulations
- ✅ Bradley-Terry preference model with base μ
- ✅ KL-regularized objectives
- ✅ Soft Bellman equations
- ✅ **Theorem: Per-step RLHF ≡ DDC ≡ MaxEnt-IRL**
- ✅ PPO as standard implementation
- ✅ Token additivity assumption
- ✅ Potential-based reward shaping

### What These Papers Add:
- 🆕 **DPO**: RL-free alternative (major gap!)
- 🆕 **Risk-aware RLHF**: CVaR objectives beyond mean reward
- 🆕 **Token-level continuous rewards**: Refines per-step formulation
- 🆕 **RLAIF**: AI-generated preferences for scalability
- 🆕 **Personalization**: Heterogeneous preferences (vs single μ)
- 🆕 **Multimodal**: Beyond text generation
- 🆕 **Safety constraints**: Helpfulness vs harmlessness trade-offs
- 🆕 **Robustness**: Handling corrupted feedback

---

## 🎯 Priority Downloads

### High Priority (Directly Relevant to Our Equivalence Theorem):
1. ✅ **Kaufmann et al. (2023)** - Already cited, comprehensive survey
2. ⭐ **RA-PbRL (2025)** - Risk-aware extension of our framework
3. ⭐ **TLCR (2024)** - Token-level continuous rewards (refines Section 5.4)
4. ⭐ **DPO Survey (2024)** - Major alternative we don't mention
5. ⭐ **RLAIF (2024)** - Scalability of Bradley-Terry data collection

### Medium Priority (Practical Extensions):
- Safe RLHF (2023) - Multi-objective constraints
- Async RLHF (2025) - Alternative to PPO
- P-ShareLoRA (2025) - Personalization methods
- Robust RLHF (2024) - Noisy preference handling

### Lower Priority (Newer Frontiers):
- MM-RLHF (2025) - Multimodal extension
- FedRLHF (2025) - Privacy-preserving
- Uni-RLHF (2024) - Benchmarking platform

---

## 📥 Download Commands

### Foundational
```bash
cd /Users/pranjal/Code/ORE/papers/RLHF/foundational/
wget https://arxiv.org/pdf/1706.03741.pdf -O christiano_2017.pdf
wget https://arxiv.org/pdf/2203.02155.pdf -O ouyang_2022_instructgpt.pdf
wget https://arxiv.org/pdf/2204.05862.pdf -O askell_2021_anthropic.pdf
```

### Surveys
```bash
cd /Users/pranjal/Code/ORE/papers/RLHF/surveys/
wget https://arxiv.org/pdf/2312.14925.pdf -O kaufmann_2023_survey.pdf
wget https://arxiv.org/pdf/2404.08555.pdf -O abdelnabi_2024_deciphered.pdf
# DPO survey - need to get PDF link from Semantic Scholar page
```

### Alternatives to RL
```bash
cd /Users/pranjal/Code/ORE/papers/RLHF/alternatives_to_RL/
wget http://arxiv.org/pdf/2405.21046.pdf -O xpo_2024.pdf
```

### Reward Modeling
```bash
cd /Users/pranjal/Code/ORE/papers/RLHF/reward_modeling/
wget https://arxiv.org/pdf/2407.16574.pdf -O tlcr_2024.pdf
wget https://arxiv.org/pdf/2411.08302.pdf -O r3hf_2024.pdf
wget https://arxiv.org/pdf/2402.09401.pdf -O dense_reward_2024.pdf
```

### Safety & Risk
```bash
cd /Users/pranjal/Code/ORE/papers/RLHF/safety_risk/
wget https://arxiv.org/pdf/2410.23569.pdf -O ra_pbrl_2025.pdf
wget https://arxiv.org/pdf/2310.12773.pdf -O safe_rlhf_2023.pdf
wget https://arxiv.org/pdf/2406.15568.pdf -O robust_rlhf_2024.pdf
wget https://arxiv.org/pdf/2503.22723.pdf -O zero_shot_llm_2025.pdf
```

### Efficiency & Scalability
```bash
cd /Users/pranjal/Code/ORE/papers/RLHF/efficiency_scalability/
wget https://arxiv.org/pdf/2410.18252.pdf -O async_rlhf_2025.pdf
wget http://arxiv.org/pdf/2309.00267.pdf -O rlaif_2024.pdf
wget https://arxiv.org/pdf/2402.00782.pdf -O active_queries_2024.pdf
wget https://arxiv.org/pdf/2402.02423.pdf -O uni_rlhf_2024.pdf
```

### Personalization
```bash
cd /Users/pranjal/Code/ORE/papers/RLHF/personalization/
wget https://arxiv.org/pdf/2408.10075.pdf -O poddar_2024.pdf
wget https://arxiv.org/pdf/2503.19201.pdf -O p_sharelora_2025.pdf
wget https://arxiv.org/pdf/2501.11463.pdf -O curiosity_driven_2025.pdf
```

### Multimodal
```bash
cd /Users/pranjal/Code/ORE/papers/RLHF/multimodal/
wget https://arxiv.org/pdf/2502.10391.pdf -O mm_rlhf_2025.pdf
```

### Federated & Privacy
```bash
cd /Users/pranjal/Code/ORE/papers/RLHF/federated_privacy/
wget http://arxiv.org/pdf/2412.15538.pdf -O fedrlhf_2025.pdf
```

---

## 📝 Notes

- All ArXiv URLs verified as of 2025-10-09
- Some papers have both HTML and PDF versions - PDF preferred for offline reading
- Semantic Scholar links may require manual download
- Check for updated versions on ArXiv (some papers have v2, v3 revisions)

---

## 🔄 Future Updates

Consider adding:
- Papers on RLHF for code generation
- RLHF for scientific reasoning
- Constitutional AI (related Anthropic work)
- Reward hacking prevention
- Iterative RLHF methods
