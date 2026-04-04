DDC and IRL Equivalence
=======================

These examples prove that Dynamic Discrete Choice estimation via NFXP and Maximum Entropy Inverse Reinforcement Learning optimize the same maximum likelihood objective. The equivalence script builds a minimal two-state MDP and shows that both methods recover identical policy parameters when rewards are anchored for identification following Kim et al. (2021). The notebooks extend the comparison to Guided Cost Learning and to a side-by-side NFXP versus MCE IRL benchmark on larger environments.

Scripts in this example directory:

- ``ddc_maxent_equivalence.py`` -- proves DDC and MaxEnt IRL equivalence on a minimal two-state MDP
- ``gcl_comparison.ipynb`` -- compares Guided Cost Learning against structural estimators
- ``nfxp_vs_mceirl_comparison.ipynb`` -- side-by-side benchmark of NFXP and MCE IRL on shared environments
