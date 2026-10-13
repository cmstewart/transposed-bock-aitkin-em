# Planned simulation

Open [planned_simulation.ipynb](../../notebooks/planned_simulation.ipynb).
It explains the design, installs a separate environment, runs a smoke test,
and provides start/resume, progress, and parameter-specific analysis cells.
The full production run is off by default.

## Starting on another computer

1. Install Python 3.11 and Git. Use a supported 64-bit CPU platform.
2. Download this repository (Code > Download ZIP) and extract it, or clone it.
3. In its root folder, run `python3.11 -m pip install jupyterlab`, then
   `python3.11 -m jupyterlab notebooks/planned_simulation.ipynb`.
   On Windows, use `py -3.11` instead of `python3.11`.
4. Run the notebook in order. Choose a durable output location before smoke
   testing if the default disk is too small. No source-code edits are needed
   for the declared design; only set the production switch when ready.

The notebook's setup creates `.simulation-env` and installs this repository's
estimator plus the baseline dependencies. It does not need private WILD data,
old study folders, API credentials, or access to the author's computer.
`torch_measure` is installed without its unrelated optional predictor stack;
the dependencies needed by the imported Rasch/2PL code are listed explicitly.
That environment is scoped to this study, not every torch_measure feature.
CPU execution is used throughout. Linux and Windows behavior must be checked
by the same smoke gate on the receiving machine; local macOS tests are not
a substitute for that check.

## Scientific protocol

The preprint plans 2,000 **assigned** replications in each of 24 conditions:
16 primary, four contaminated-normal generation conditions, and four clustered
item conditions. This is 48,000 matrices and 184,000 applicable fitting attempts.
All fitted transposed populations remain normal. Only 1PL/2PL are fitted;
TBAEM-C is omitted for 1PL because it is not a distinct model there.

The starting seed is new. Results must not be pooled with either the original
100-replication comparison or the old 200-attempt protocol. Changing the design,
replication budget, source code or numerical environment after freezing requires
a separate study and output tree, not overwriting a lock.

`design.py`, `recovery.py`, the baseline adapters, and the analysis are adapted
from the prior local confirmation/validation code. The runner no longer imports
that private tree. `fit.py` calls the public `LaplaceIRT` API, using explicit
posterior means for item recovery rather than its default item-mode table.
It can also be called with an empirical training matrix and holdout mask.
This does not rerun the WILD application, whose retained fits used version 1.3.1.

### Numerical settings

- TBAEM: normalized Laplace moment map, float64, damping 0.5, at most 1,000
  iterations; the public package's fixed-point rule is not a score-root proof.
- JML: torch_measure Rasch/TwoPL, float64 LBFGS with strong-Wolfe search,
  regularization 0.01, at most 300 optimizer calls, objective/gradient checks.
- VI: py-irt 0.7.1 likelihood and mean-field guide, regular priors (SD theta 1,
  b 1.5, log(a) 0.5), at most 4,000 epochs, training-probe/parameter stability.
- Common generated data and holdout, but method-specific point estimates,
  priors/penalties and numerical criteria. This is not equal optimization effort
  or a prior-matched causal comparison.

The adapter retains all returns, including iteration limits. JML/VI and TBAEM
use different prediction estimators, as described in the notebook. Native and
identified parameter scales are recorded. Population errors in the
misspecification study are generating-moment discrepancies, not bias about an
established likelihood target. No parameter standard errors or interval
coverage are produced. The existing WILD predictive bootstrap is not rerun.

## Running outside Jupyter

After notebook setup, use `.simulation-env/bin/python` (Windows:
`.simulation-env/Scripts/python.exe`) for the commands below:

```sh
python studies/confirmation/launch.py smoke
python studies/confirmation/launch.py check
python studies/confirmation/launch.py run --workers 88
python studies/confirmation/launch.py status
python studies/confirmation/launch.py analyze
python studies/confirmation/launch.py analyze --require-complete
```

Supply `--output PATH` consistently to use another destination. To stop after
a bounded batch, use `run --max-matrices 24`; it does not lower the 2,000 target.
Create `OUTPUT/STOP` for a graceful stop after the current matrices. Remove it
to resume. Keep the computer awake.

`run --workers N` fits N matrices at once on one machine, each in its own
single-threaded process (default 1). Seeds and estimators do not depend on N.
On Linux each worker's fits are pinned to one core, because XLA otherwise
spreads large TBAEM fits over several cores despite the BLAS thread limits.
Use at most one worker per physical core, or recorded runtimes are inflated.
Each session's worker count is appended to `execution_log.jsonl`. Only one
supervisor writes each output tree; do not use a cloud-synchronized live
directory as a multi-machine work queue.

The runner retains failed and interrupted attempts, does not replace their
seeds, and refuses changed configurations. Source hashes, numerical dependency
versions, Python version and platform are recorded. Exact numerical equality
across hardware is not guaranteed. The storage gate is a planning allowance,
not a disk quota guarantee. Check available space during long runs.
For all 48,000 matrices, the initial planning allowance is about 1,427 GiB
(1.4 TiB) of free space. Compressed outputs can be smaller, but the gate does
not assume a compression ratio. Choose a sufficiently large output drive
before freezing the run; runtime has not been benchmarked at production sizes.

## Results

Every fit saves a terminal summary; returned fits also retain truth/estimate/error
vectors for theta, b, log(a), and a, predictions, histories, and applicable
population/posterior/guide exports. The analysis rechecks these arrays before
writing summaries. It separates parameters, fixed versus estimated quantities,
all-returned versus numerically stable cohorts, and paired denominators.

Monte Carlo standard errors use independent replications. Average centered
b/log(a) bias may be zero by construction. `completion_manifest.json` means
all assigned attempts have terminal records, not that all estimates converged.
Empty or incomplete outputs must not be described as finished evidence.
