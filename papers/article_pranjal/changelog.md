# Changelog - Working Version Edits

## 2026-01-17: AIRL Extensions and New Citations

### Papers Downloaded
Downloaded 5 papers to `new_papers_to_add/`:
1. `liang2026_airl_travel.pdf` - Liang et al. (2026) "Analyzing sequential activity and travel decisions with interpretable deep inverse reinforcement learning"
2. `khwaja2026_rl_ccs.pdf` - Khwaja & Srivastava (2026) "RL Based Computationally Efficient CCS Estimation of DDC Models"
3. `lee2025_serialized_content_airl.pdf` - Lee, Sudhir, Wang (2026) "Modeling Serialized Content Consumption: Adversarial IRL for Dynamic Discrete Choice"
4. `kaji2023_adversarial_estimation.pdf` - Kaji, Manresa, Pouliot (2023) "An Adversarial Approach to Structural Estimation"
5. `zhao2023_deep_irl_route.pdf` - Zhao & Liang (2023) "A deep IRL approach to route choice modeling"

All papers transcribed to markdown via docling.

---

### New Citations Added to `our-cites.bib`

#### kaji:2023
```bibtex
@article{kaji:2023,
author = {Tetsuya Kaji and Elena Manresa and Guillaume Pouliot},
title = {An Adversarial Approach to Structural Estimation},
journal = {Econometrica},
volume = {91},
number = {6},
year = {2023},
pages = {2041--2063}
}
```

#### lee:2026
```bibtex
@unpublished{lee:2026,
author = {Peter S. Lee and K. Sudhir and Tong Wang},
title = {Modeling Serialized Content Consumption: Adversarial IRL for Dynamic Discrete Choice},
note = {Job Market Paper, Yale School of Management},
year = {2026}
}
```

---

### Edits to IRL.tex

#### 1. Kaji et al. paragraph (line 58)
**Added after AIRL paragraph:**
> The adversarial framework has a direct analog in structural econometrics. \citet{kaji:2023} formalize this connection: a discriminator trained to distinguish real from simulated data automatically selects optimal moments. With a logistic discriminator this reduces to optimally-weighted simulated method of moments (SMM); with a neural network it approximates MLE without a closed-form likelihood.

#### 2. Lee et al. heterogeneity (line 58, continued)
**Added:**
> \citet{lee:2026} extend AIRL to handle unobserved heterogeneity via a posterior network that assigns users to latent segments, avoiding intractable EM algorithms.

#### 3. Lee et al. exit anchor proof (line 56 footnote)
**Added to AIRL footnote:**
> However, \citet{lee:2026} prove that action-dependent rewards are identified within AIRL when constrained by an exit anchor.

#### 4. Lee et al. text data (line 223)
**Added after Barnes/Zhao applications:**
> \citet{lee:2026} extend this to unstructured text data.

---

### Summary of Lee et al. (2026) Citations

| Location | Contribution |
|----------|--------------|
| IRL.tex:56 footnote | Exit anchor identification proof |
| IRL.tex:58 | Posterior network for heterogeneity |
| IRL.tex:223 | Unstructured text/LLM state spaces |

---

---

### Liang et al. (2025) Citations

Added `liang:2025` to bib (Yuebing Liang et al. - "Analyzing sequential activity and travel decisions with interpretable deep inverse reinforcement learning" - Travel Behaviour and Society).

| Location | Text Added |
|----------|------------|
| IRL.tex:223 | "In a recent extension, \citet{liang:2025} bridge the gap to structural interpretation via knowledge distillation: they train a surrogate multinomial logit model on the probabilities of the Deep AIRL policy, recovering interpretable preference parameters." |
| Conclusion.tex:13 | "However, emerging frameworks like \citet{liang:2025} attempt to resolve this by projecting high-dimensional Deep IRL policies onto parsimonious surrogate models to recover meaningful behavioral insights." |

**Rationale:**
- IRL.tex:223 - Groups with zhao:2023 (same research team), shows evolution from application to structural interpretation
- Conclusion.tex:13 - Counter-example to "black box" criticism of IRL

---

### Files Modified
- `IRL.tex` - 5 additions (Kaji paragraph + 3 Lee citations + 1 Liang citation)
- `Conclusion.tex` - 1 addition (Liang citation)
- `our-cites.bib` - 3 new entries (kaji:2023, lee:2026, liang:2025)

### Files Created
- `new_papers_to_add/` - 5 PDFs + 5 markdown transcriptions
- `changelog.md` - this file
