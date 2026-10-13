#!/usr/bin/env python3
"""Run one coordinate, with immutable outcomes and a separate smoke namespace."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Pin to the core launch.py assigned before numerical libraries start thread pools.
# XLA otherwise spreads large TBAEM fits over several cores despite the BLAS limits.
if "TBAEM_CPU" in os.environ and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {int(os.environ["TBAEM_CPU"])})

from protocol_tools import ROOT, OUTPUT, protocol, verify
sys.path.insert(0, str(ROOT))
from design import Cell, cells, generate, methods, population_truth
from fit import fit_one
from protocol_tools import write_json
from recovery import identify, probability_invariance, save_recovery
from filelock import FileLock

import numpy as np
from scipy.special import expit


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path), timeout=0):
        yield


def run_cell(cell, settings, output, source_hash, phase):
    directory = output / "recovery" / cell.label / f"rep_{cell.replicate:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    configuration = {"protocol": settings, "cell": asdict(cell), "phase": phase,
                     "source_manifest_sha256": source_hash}
    config_path = directory / "configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise RuntimeError(f"Refusing a changed configuration: {directory}")
    else:
        write_json(config_path, configuration)
    data = generate(cell, settings)
    truths, truth_transform = identify(data["theta"] + data["b"].mean(), data["b"], np.log(data["a"]), cell.model)
    np.testing.assert_allclose(truths["theta"], data["theta"], atol=1e-12)
    input_path = directory / "data.npz"
    retained = {k: v for k, v in data.items() if k != "generating_probability"}
    if input_path.exists():
        with np.load(input_path, allow_pickle=False) as archive:
            assert set(archive.files) == set(retained)
            for key, value in retained.items():
                np.testing.assert_array_equal(archive[key], value)
    else:
        with input_path.open("xb") as stream:
            np.savez_compressed(stream, **retained)
    rows, cols = np.where(data["holdout"])
    for method in methods(cell):
        destination = directory / method
        summary_path = destination / "summary.json"
        if summary_path.exists():
            saved = json.loads(summary_path.read_text())
            assert saved["cell"] == asdict(cell) and saved["method"] == method
            assert saved["source_manifest_sha256"] == source_hash and saved["phase"] == phase
            print(f"[retained] {cell.label} rep={cell.replicate} {method}", flush=True)
            continue
        interrupted = destination.exists() and any(destination.iterdir())
        destination.mkdir(exist_ok=True)
        summary = {"cell": asdict(cell), "cell_label": cell.label, "method": method,
                   "phase": phase, "source_manifest_sha256": source_hash}
        if interrupted:
            summary.update(status="interrupted", reason="Unsummarized prior attempt retained; seed is not retried", elapsed_seconds=None)
            write_json(summary_path, summary)
            continue
        write_json(destination / "started.json", {"time": time.time(), "seed": cell.seed})
        started = time.perf_counter()
        try:
            fit = fit_one(data, cell.model, method, cell.seed, settings, destination / "progress.json")
            estimates, transform = identify(fit["theta"], fit["b"], fit["log_a"], cell.model)
            metrics = save_recovery(destination / "parameters.npz", estimates, truths,
                {"cell": asdict(cell), "method": method, "transform": transform,
                 "truth_transform": truth_transform, "role_a_1pl": "fixed_not_estimated",
                 "point_estimate": fit.get("item_point_estimate", "package_parameter_export")})
            export_error = probability_invariance(fit["theta"], fit["b"], fit["log_a"], estimates, rows, cols)
            probability = fit.get("probability")
            if probability is None:
                probability = expit(np.exp(fit["log_a"][cols]) * (fit["theta"][rows] - fit["b"][cols]))
            if probability.shape != rows.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError("Invalid held-out probability export")
            if export_error > 1e-10:
                raise ValueError("Identification changed response probabilities")
            y = data["response"][rows, cols]
            clipped = np.clip(probability, 1e-12, 1 - 1e-12)
            summary.update(status="returned", metrics=metrics, diagnostics=fit["diagnostics"],
                           identification_probability_error=export_error,
                           brier=float(np.mean((probability - y)**2)),
                           logloss=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))))
            if "population" in fit:
                summary["population_reference"] = ("generating_moment_discrepancy_not_likelihood_target_bias"
                    if cell.condition == "contaminated_normal" else "generating_population")
                summary["population_metrics"] = save_recovery(destination / "population.npz",
                    fit["population"], population_truth(cell), {"target": summary["population_reference"],
                    "fixed_zero": ["cov_b_log_a", "corr_b_log_a"] if method == "tbaem_d" else []})
                np.savez_compressed(destination / "posterior.npz", modes=fit["posterior_modes"], covariance=fit["posterior_covariance"])
            if "guide" in fit:
                np.savez_compressed(destination / "guide.npz", **fit["guide"])
            np.savez_compressed(destination / "predictions.npz", rows=rows, cols=cols, probability=probability, y=y)
            write_json(destination / "history.json", fit["history"])
        except Exception as error:
            summary.update(status="failed", reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(summary_path, summary)
        print(f"[{summary['status']}] {cell.label} rep={cell.replicate} {method} "
              f"{summary['elapsed_seconds']:.1f}s "
              f"{summary.get('diagnostics', {}).get('stopping_reason', summary.get('reason'))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--smoke-coordinate", type=Path)
    args = parser.parse_args()
    phase = "smoke" if args.smoke_coordinate else "production"
    source_hash = verify(args.output.parent if phase == "smoke" else args.output)
    settings = protocol()
    if phase == "smoke":
        cell = Cell(**json.loads(args.smoke_coordinate.read_text()))
        if cell.label != args.cell or cell.replicate != args.replicate:
            raise ValueError("Smoke coordinate does not match command")
        if args.output.name != "smoke":
            raise ValueError("Smoke output must remain in the isolated smoke tree")
    else:
        from launch import check_gate
        check_gate(args.output)
        candidates = [c for c in cells(settings) if c.label == args.cell and c.replicate == args.replicate]
        if len(candidates) != 1:
            raise ValueError("Unknown production coordinate")
        cell = candidates[0]
    # One process per matrix; launch.py's supervisor lock excludes a second supervisor.
    with exclusive_lock(args.output / "locks" / cell.label / f"rep_{cell.replicate:03d}.lock"):
        run_cell(cell, settings, args.output, source_hash, phase)


if __name__ == "__main__":
    main()
