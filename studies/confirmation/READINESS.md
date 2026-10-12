# Execution checks

Checked on October 12, 2026, using macOS arm64 and a new Python 3.11.15
environment. No production replications were started.

- Fresh installation imported the public estimator, torch_measure and py-irt.
- All 49 tests passed: 33 estimator tests and 16 study tests.
- The notebook executed in Jupyter with its production switches off.
- All 24 full-size data generators passed the holdout and observation checks.
- Six small matrices produced 23 applicable fits using full optimizer budgets.
  All 23 returned; 20 met their numerical stopping rules and three reached
  iteration limits. No software failures or interrupted attempts occurred.
- Saved parameter vectors and predictions passed the analysis audit.
- Tests check unique deterministic production seeds, parameter-scale invariance,
  population moments, Monte Carlo errors, failure preservation, source-change
  refusal, duplicate-worker exclusion and detection of corrupted summaries.

The smoke fits are not production evidence and do not establish statistical
validity, convergence at production sizes, or a full-study runtime. Linux and
Windows have not been tested here; run the notebook's smoke gate on the actual
machine before production. The initial free-space planning allowance is about
1,427 GiB. Compressed usage may be lower, but sufficient durable storage and
regular backups are still required.

The current manuscript's 100-replication comparison remains separate. This
runner implements the planned 2,000 assigned replications per condition and
does not replace or relabel the earlier results.
