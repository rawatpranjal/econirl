# ReadTheDocs Documentation Design

**Date:** 2026-01-18
**Status:** Approved
**Audience:** Industry practitioners (like DoubleML/EconML)

## Overview

Add Sphinx-based documentation hosted on ReadTheDocs with tutorial-focused content for industry practitioners.

## Project Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst               # Landing page
├── installation.rst        # Installation guide
├── quickstart.rst          # 5-minute getting started
├── tutorials/
│   ├── index.rst
│   ├── rust_bus_replication.ipynb    # Tutorial 1
│   ├── simulated_data_workflow.ipynb # Tutorial 2
│   └── real_data_example.ipynb       # Tutorial 3
├── api/
│   ├── index.rst
│   ├── environments.rst
│   ├── estimation.rst
│   ├── inference.rst
│   └── ...
├── _static/                # CSS, images
└── _templates/             # Custom templates
.readthedocs.yaml           # RTD build config
```

## Tooling

- **sphinx-autodoc** - Auto-generate API docs from docstrings
- **nbsphinx** - Render Jupyter notebooks as tutorial pages
- **pydata-sphinx-theme** - Clean, modern theme (pandas, numpy, scikit-learn)
- **sphinx-copybutton** - One-click code copying
- **myst-parser** - Write docs in Markdown where convenient

## Core Pages

### Landing Page (index.rst)
- Tagline: "The StatsModels of IRL"
- Feature highlights in 3-4 bullet cards
- Code snippet showing complete workflow in ~10 lines
- Links to Installation → Quick Start → Tutorials

### Installation Page
- Primary: `pip install econirl`
- Development: `pip install -e ".[dev]"`
- Optional dependencies explained
- Python ≥3.10 requirement

### Quick Start Page
- Expands README example with brief explanations
- Three steps: Define environment → Simulate/load data → Estimate and interpret
- Shows `result.summary()` output
- "Next steps" pointing to tutorials

## Tutorials

### Tutorial 1: Rust Bus Replication
- **Goal:** Reproduce Rust (1987) results using original data
- **Sections:**
  - Background (2 paragraphs max)
  - Load Rust's original data
  - Define environment and utility specification
  - Estimate with NFXP, show `summary()`
  - Compare to Rust's published results
  - Economic interpretation
- **Outcome:** Practitioners see validated, credible results

### Tutorial 2: Simulated Data Workflow
- **Goal:** Show parameter recovery and validation workflow
- **Sections:**
  - Set "true" parameters for synthetic environment
  - Generate panel data with `simulate_panel()`
  - Estimate parameters
  - Compare estimated vs true - visualize recovery
  - Discuss sample size effects, identification
- **Outcome:** Practitioners understand validation workflow

### Tutorial 3: Real Data Example
- **Goal:** Apply to different public dataset beyond Rust
- **Candidates:** Kennan (1985) job search, Pakes (1986) patents, transportation mode choice
- **Workflow:** Load → specify → estimate → interpret
- **Outcome:** Practitioners see econirl generalizes

## API Documentation

### Organization (by module)
- `econirl.environments` - RustBusEnvironment, base classes
- `econirl.estimation` - NFXPEstimator, base classes
- `econirl.inference` - Results, standard errors, identification
- `econirl.preferences` - LinearUtility, base classes
- `econirl.simulation` - simulate_panel, counterfactual
- `econirl.visualization` - policy plots, value function plots

### Docstring Standard: NumPy style
```python
def estimate(self, data, utility, spec, transitions):
    """Estimate structural parameters using NFXP.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with columns ['id', 'period', 'state', 'action']
    utility : BaseUtility
        Utility specification with parameters to estimate

    Returns
    -------
    EstimationResult
        Results object with .summary(), .params, .std_errors
    """
```

## Build & Deployment

### pyproject.toml additions
```toml
[project.optional-dependencies]
docs = [
    "sphinx>=7.0",
    "pydata-sphinx-theme>=0.14",
    "nbsphinx>=0.9",
    "sphinx-copybutton>=0.5",
    "myst-parser>=2.0",
    "ipykernel",
]
```

### .readthedocs.yaml
- Build on Python 3.11
- Install package with `[docs]` dependencies
- Optional PDF build

### Hosting
- Connect GitHub repo to ReadTheDocs (free for open source)
- Auto-build on push to `main`
- Version dropdown for releases

## Data Bundling

- Rust's original bus data (~50KB) bundled in `econirl/data/`
- Accessible via `econirl.datasets.load_rust_bus()`
- Keeps tutorials self-contained and reproducible

## Style Guidelines

- Lead with code, explain after
- Avoid mathematical notation on main pages
- Every code block copy-paste runnable
- Practical language: "estimate parameters" not "solve the inverse problem"

## Next Steps

1. Set up Sphinx scaffolding and configuration
2. Create core pages (index, installation, quickstart)
3. Write Tutorial 1: Rust Bus Replication
4. Write Tutorial 2: Simulated Data Workflow
5. Research and write Tutorial 3: Real Data Example
6. Generate API documentation from docstrings
7. Configure ReadTheDocs and deploy
