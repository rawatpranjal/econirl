# Estimators

Estimator pages are the active RTD surface. Each page is a short front door to
the estimator-specific reference PDF. The PDFs are built from dedicated TeX
files and use the shared known-truth synthetic DGP harness rather than real
data examples.

NFXP, CCP, MPEC, and SEES are fully migrated estimator pages and PDFs. NNES is
wired into the same harness but currently reports a failing diagnostic run. The
remaining estimator pages are kept compact while their known-truth PDFs are
migrated.

```{toctree}
:caption: Structural Econometrics
:maxdepth: 1

estimators/nfxp
estimators/ccp
estimators/mpec
estimators/sees
estimators/nnes
estimators/tdccp
```

```{toctree}
:caption: Inverse Reinforcement Learning
:maxdepth: 1

estimators/mce_irl
estimators/airl
estimators/airl_het
estimators/f_irl
estimators/gladius
estimators/iq_learn
```
