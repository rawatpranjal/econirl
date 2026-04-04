Ziebart MCE IRL Replication
===========================

These examples replicate the gridworld experiments from Ziebart (2008, 2010) using econirl. The replication script recovers reward parameters on a five-by-five gridworld with MCE IRL and compares the results against MaxEnt IRL to show where the causal entropy formulation improves over the non-causal version. The gridworld benchmark script extends the comparison across three reward specifications and evaluates in-sample, out-of-sample, and transfer performance for MCE IRL, MaxEnt IRL, and NFXP.

Scripts in this example directory:

- ``ziebart_mce_irl_replication.py`` -- replicates the Ziebart gridworld experiment with reward recovery and policy comparison
- ``run_gridworld.py`` -- benchmarks MCE IRL, MaxEnt IRL, and NFXP across three reward cases with full evaluation
