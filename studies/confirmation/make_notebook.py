"""Build the checked-in notebook from short, reviewable cells."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(True)})


md("""# Planned transposed Bock-Aitkin EM simulation

This notebook runs the follow-up described in the preprint, not the earlier
100-replication comparison. It uses the public GitHub estimator and a separate
output directory. **2,000 assigned replications per condition, 24 conditions,
48,000 response matrices, 184,000 applicable fits.** Failures count as attempts;
we do not replace them to obtain 2,000 successes.

Use a **Python 3.11 kernel**, Git, and a machine that can stay awake for a long
CPU run. Installation needs internet; simulation data are generated locally.
The full study is not a short laptop demonstration. A disposable Colab runtime
is unsuitable for the complete run. Keep outputs on a durable local or mounted
drive with substantial free space, and back them up after stopping the worker.

Run the cells in order. The full study is **off by default**. The setup installs
an isolated environment rather than replacing packages in your notebook kernel.
""")
code('''from pathlib import Path
import json
import os
import subprocess
import sys

# Works when opened from the repository root or its notebooks folder.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "studies/confirmation/launch.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Extract the supplied repository ZIP first, then open its notebooks folder in Jupyter.")
STUDY = ROOT / "studies" / "confirmation"
OUTPUT = Path(os.environ.get("TBAEM_OUTPUT", ROOT / "simulation-output")).expanduser().resolve()
ENVIRONMENT = Path(os.environ.get("TBAEM_SIM_ENV", ROOT / ".simulation-env")).expanduser().resolve()
PYTHON = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
print("Repository:", ROOT)
print("Results:", OUTPUT)
''')
md("""## 1. Install and check the three frameworks

Download the repository from https://github.com/pbotter42/transposed-bock-aitkin-em
(Code > Download ZIP), extract it, and open this notebook. No private folders,
API keys, WILD data, or hard-coded user paths are needed. Keep this source
snapshot unchanged once the study starts.

TBAEM comes from this repository. `torch_measure` is pinned to the commit used
for the baseline adapters, and `py-irt` to version 0.7.1. Only the imports needed
by the Rasch/2PL study are installed; unrelated torch_measure predictors are not
supported by this study environment. Installation can take several minutes.
""")
code('''subprocess.run([sys.executable, str(STUDY / "setup.py"),
                "--environment", str(ENVIRONMENT)], check=True)
def study(action, *options):
    return subprocess.run([str(PYTHON), str(STUDY / "launch.py"), action,
                           "--output", str(OUTPUT), *map(str, options)], check=True)
subprocess.run([str(PYTHON), "-m", "pytest", str(STUDY / "test_study.py"), "-q"], check=True)
''')
md("""## 2. What are the conditions?

Rows are subjects (here, AI systems); columns are binary items. All capabilities
are generated from a standard normal distribution. Item difficulty SD is 0.8
and log-discrimination SD is 0.35. There is no additional missingness; a common
10% random-cell holdout evaluates predictions for the same subjects and items.

| Study | Conditions | Design |
|---|---:|---|
| Primary | 16 | 1PL and 2PL; n = 20/100; p = 1,000/10,000; 2PL correlation = -0.4/0/0.4 |
| Item-population misspecification | 4 | 2PL, correlation 0.4; 10% inflated normal item vectors, variance rescaled |
| Dependent items | 4 | 2PL, correlation 0.4; clusters of 20 items, shared variance fraction 0.3 |

The last two studies use the same n and p grid. They change **generation**, not
the fitted family. Every TBAEM fit has a normal item population and uses Laplace
integration. No 3PL, 4PL, alternative fitted distributions, or 40% missingness
conditions are included. Realized log discriminations are centered before
generating responses; finite-bank item draws are consequently coupled.
""")
code('''settings = json.loads((STUDY / "protocol.json").read_text())
print(json.dumps(settings, indent=2))
study("status")  # A separate line for every condition, including zero-count cells.
''')
md("""## 3. What gets fitted?

| Method | Framework | What is estimated |
|---|---|---|
| TBAEM-D | This GitHub package | Capabilities and a diagonal normal item population |
| TBAEM-C | This GitHub package | The 2PL also estimates correlation between b and log(a) |
| JML | torch_measure | Joint subject/item parameters with the declared regularization |
| VI | py-irt | Mean-field variational parameters with explicit item priors |

TBAEM-C is not a distinct 1PL model, so it is omitted for 1PL. The baseline
adapters use the WILD-style settings: float64 JML with a gradient check and
up to 300 LBFGS calls; VI with up to 4,000 epochs and priors SD(theta)=1,
SD(b)=1.5 and SD(log(a))=0.5. TBAEM has a 1,000-iteration safety limit,
damping 1/2 and consecutive fixed-point checks. Equal iteration counts would
not mean equal computational effort. Numerical eligibility is method-specific.

The public TBAEM release is 1.4.0rc1; the reported WILD fits used frozen 1.3.1.
Regression tests compare the normalized map, but this study does not relabel
old WILD results or claim all edge cases are identical. Future empirical fits
can call the same `fit.py` adapter. No WILD fitting is performed here.

TBAEM item recovery uses E(b), E(log(a)), and exp(E(log(a))), not E(a).
TBAEM prediction averages response probabilities over posterior cubature;
JML and VI use their declared point exports. See the source adapters for the
exact objectives, initialization, and stopping diagnostics.
""")
md("""## 4. Smoke test before production

This checks all 24 full-size generators and fits six small response matrices
covering the primary correlations and both sensitivities. It uses the full
optimizer budgets and every applicable method, then audits the saved parameter
vectors and predictions. Small-sample iteration limits are not software failure.
This is an operational check, not validation of statistical correctness or an
estimate of runtime at the largest sample size.

The smoke test freezes sources, settings and environment versions in the output
directory. Its seeds and files are separate from production. If it fails, inspect
`smoke/run.log`; do not delete outcomes or change the frozen run to force a pass.
""")
code('''study("smoke")
''')
md("""## 5. Check storage, then start deliberately

The storage gate uses an uncompressed-array planning allowance plus trace space
and headroom. It refuses a run when the destination is too small. This estimate
is conservative, not a guaranteed upper bound. Runtime depends strongly on
hardware and conditions; the smoke test does not provide a reliable full-study ETA.
The full-run allowance is about **1,427 GiB (1.4 TiB) free**. Use an adequately
sized drive before freezing the study; compression may reduce actual usage.

Set `RUN_PRODUCTION = True` only when you are ready. `MATRICES_THIS_SESSION = 0`
means work toward all 48,000 matrices. A positive value stops after that many
new matrices without changing the 2,000-per-condition target. Each matrix runs
in a fresh single-threaded process; all conditions advance together.

`WORKERS` sets how many matrices are fitted at once. Seeds, estimators and
settings do not depend on it, so it changes wall-clock time, not estimates.
On Linux each worker's fits are pinned to their own core, so every method is
single-threaded; use at most one worker per physical core, or recorded per-fit
runtimes are inflated. On one worker the full study takes
roughly 80 days; 88 workers take roughly one day. Each session's worker count
is recorded in `execution_log.jsonl`.
""")
code('''RUN_PRODUCTION = False
MATRICES_THIS_SESSION = 0
WORKERS = 1
if RUN_PRODUCTION:
    study("check")
    study("run", "--max-matrices", MATRICES_THIS_SESSION, "--workers", WORKERS)
else:
    print("Production is off. No production fits were launched.")
''')
md("""## 6. Progress, stopping and resuming

The run cell stays busy and prints each matrix as it starts. For progress from
another terminal, run the command printed below. Per-fit details are appended
to `run.log` (one file per worker under `logs/` when `WORKERS > 1`); TBAEM
also writes `progress.json` inside its fit folder.

To stop safely, create an empty file named `STOP` inside the output directory.
The supervisor stops after the current matrices finish. Remove that file and rerun the
run cell to resume. Do not start a second notebook on the same outputs.
If the machine or process is interrupted mid-fit, that attempt is recorded as
interrupted on resume and is not fitted again. Finished failures and limits
are retained, not replaced. Copy the entire output tree, including the lock,
when backing it up. Cross-platform bitwise equality is not promised.
""")
code('''import shlex
print(shlex.join([str(PYTHON), str(STUDY / "launch.py"), "status", "--output", str(OUTPUT)]))
study("status")
''')
md("""## 7. Read parameter-specific results

The analysis audits saved vectors before writing CSVs. It reports bias, MAE,
MSE, pooled RMSE, mean replication RMSE, rank correlation and Monte Carlo
standard errors **separately** for theta, b, log(a), a, and population quantities.
Truth-bin errors and paired method contrasts retain their actual denominators.
All-returned and rule-stable cohorts are separate. Centered b/log(a) mean bias
can be zero by construction; that is not evidence of unbiased estimation.

MCSE describes simulation precision across independent replications. It is
not a fitted parameter's standard error. No parameter intervals, coverage or
new bootstrap study are claimed. Sensitivity population errors are differences
from generating moments, not bias about a proven pseudo-true target.

The files are `completion_by_cell_method.csv`, `attempts.csv`, `performance.csv`,
`parameter_recovery.csv`, `population_recovery.csv`, `truth_bin_recovery.csv`,
and `paired_contrasts.csv`. Raw inputs, truths, all parameter estimates,
histories and held-out predictions remain under `recovery/`.
""")
code('''SUMMARIZE_PRODUCTION = False
if SUMMARIZE_PRODUCTION:
    study("analyze")  # Clearly labels incomplete results.
    import csv
    with (OUTPUT / "analysis/completion_by_cell_method.csv").open() as stream:
        for row in csv.DictReader(stream):
            print(row)
else:
    print("No production report requested.")
''')
md("""## 8. Final completion gate

This succeeds only when every assigned method attempt has a terminal record.
It does not require every fit to succeed. Do not combine this study with the
100-replication preprint results or the earlier 200-attempt protocol.
""")
code('''CHECK_FINAL_COMPLETION = False
if CHECK_FINAL_COMPLETION:
    study("analyze", "--require-complete")
''')

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[2] / "notebooks/planned_simulation.ipynb"
    for i, cell in enumerate(cells):
        cell["id"] = f"study-{i:02d}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    destination.write_text(json.dumps(notebook, indent=1) + "\n")
    print(destination)
