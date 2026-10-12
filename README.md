# Transposed Bock-Aitkin EM

**Software for the September 2026 preprint, not the final manuscript.**
The method and its evaluation remain under development. This release does
not provide validated parameter standard errors or confidence intervals.

This package fits binary item response models when there are many items and
relatively few subjects, such as language models answering benchmark questions.
It treats subjects' capabilities as fixed parameters and integrates over a
normal population of item effects. This reverses the usual Bock-Aitkin
integration over subjects' capabilities.

The implementation uses a Laplace approximation and a damped, normalized
posterior-moment iteration. It is **not an exact marginal-likelihood optimizer**.
The [method notes](docs/method.md) describe the equations, identification,
updates, and stopping rule.
Read the [numerical limitations](docs/method.md#numerical-limitations-in-140rc1)
before using extreme-response or frozen-bank scores. The preprint update does
not change the fitting code or imply that these edge cases are fixed.

Supported models are the 1PL and the 2PL, with either zero or estimated
correlation between item difficulty `b` and **log discrimination** `log(a)`.
There is no 3PL, 4PL, or parameter-inference API.

## Install

Use Python 3.10-3.12 (3.11 recommended) in a new virtual environment.
From a terminal with that environment active:

```sh
python -m pip install "git+https://github.com/pbotter42/transposed-bock-aitkin-em.git@v1.4.0rc1"
```

No private files, data downloads, API keys, or machine-specific paths are needed.
Dependencies are installed automatically. JAX requires a supported platform.
The release is tested locally on macOS with Python 3.11; a Linux test workflow
is configured for Python 3.10-3.12. See [release checks](docs/release-checks.md)
for what was actually run and the current GitHub Actions limitation.

## Fit a Model

```python
import numpy as np
from scipy.special import expit
from transposed_em import LaplaceIRT

rng = np.random.default_rng(42)
theta = np.linspace(-1.5, 1.5, 12)
b = rng.normal(size=80)
a = np.exp(rng.normal(0, 0.3, size=80))
responses = rng.binomial(1, expit(a * (theta[:, None] - b)))

fit = LaplaceIRT(model="2pl", covariance="full", max_iterations=40).fit(responses)
print(fit.subject_frame())
print(fit.item_frame())
print(fit.summary())
```

Rows are subjects or systems; columns are items. Responses must be `0`, `1`,
`NaN`, or `-1` (missing). A pandas DataFrame retains row and column labels.
Use `model="1pl"` for fixed discrimination, or `covariance="diagonal"` for the
2PL with zero population correlation. Computation uses 64-bit arithmetic.

The short example may reach its iteration limit. Check `stopping_reason_`
and `diagnostics_` before using estimates. The default limit is 1,000;
`fixed_point_criteria` means the numerical map was stable, not that exact
likelihood equations were solved. Item exports are conditional modes, not
posterior means. See the [API guide](docs/api.md) for prediction and new-subject
scoring, and the [reproducibility notes](docs/reproducibility.md) for the
relationship between this release and the reported analyses.

## Paper and Results

The [preprint release](https://github.com/pbotter42/transposed-bock-aitkin-em/releases/tag/preprint-v29)
contains the main PDF, supplement, and an Overleaf-ready source ZIP.
The supplement contains WILD capability estimates and a guide to the
condition-specific simulation result files supporting the current article.
The [`results/`](results) directory contains the preprint's saved result tables.
The article reports a 100-replication comparison and the WILD application.
The planned study has **2,000 assigned replications per condition**
across 24 conditions: 48,000 datasets and 184,000 applicable fits. It is future
work, not a result of this release.
The [planned-simulation notebook](notebooks/planned_simulation.ipynb) provides
the separate 2,000-replication runner, setup, smoke test, resume controls, and
parameter-specific reports. See its [run guide](studies/confirmation/README.md)
before starting; the full study is off by default and is computationally large.
The repository does not redistribute WILD responses or claim
to reconstruct individual parameter vectors that were not saved for the
100-replication comparison.

For development, clone this repository, run `python -m pip install -e '.[dev]'`,
then `pytest -q`. Tests include numerical agreement with the frozen diagnostic
implementation for all three model specifications. They do not establish
statistical validity or replace the paper's planned studies.

Code is MIT licensed. Citation information is in [CITATION.cff](CITATION.cff).
The preprint discloses AI assistance in code development and preparation;
the authors remain responsible for the work.
