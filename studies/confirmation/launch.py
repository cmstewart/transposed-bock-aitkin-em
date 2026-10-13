"""Smoke, resume, inspect, and summarize the 24-condition study."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

from design import Cell, cells, generate, methods, seed_for
from protocol_tools import STUDY, OUTPUT, freeze, protocol, verify, write_json
from run import exclusive_lock


def expected():
    return sorted(cells(protocol()), key=lambda c: (c.replicate, c.label))


def directory(output, cell):
    return output / "recovery" / cell.label / f"rep_{cell.replicate:03d}"


def done(output, cell):
    return all((directory(output, cell) / m / "summary.json").exists() for m in methods(cell))


def status(output):
    from collections import Counter
    groups = {}
    for cell in expected():
        row = groups.setdefault(cell.label, {"condition": cell.label, "assigned": 0,
                                             "finished_matrices": 0, "recorded_fits": 0,
                                             "returned": 0, "failed": 0, "interrupted": 0})
        row["assigned"] += 1
        row["finished_matrices"] += int(done(output, cell))
        for method in methods(cell):
            path = directory(output, cell) / method / "summary.json"
            if path.exists():
                value = json.loads(path.read_text())
                if value["cell"] != asdict(cell) or value["method"] != method:
                    raise ValueError(f"Coordinate mismatch: {path}")
                row["recorded_fits"] += 1
                row[value["status"]] += 1
    return list(groups.values())


def cores():
    """CPUs available to this process; each worker's fits are pinned to one of them on Linux."""
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []


def child(cell, output, smoke_coordinate=None, log_path=None, worker=0):
    command = [sys.executable, str(STUDY / "run.py"), "--cell", cell.label,
               "--replicate", str(cell.replicate), "--output", str(output)]
    if smoke_coordinate:
        command += ["--smoke-coordinate", str(smoke_coordinate)]
    environment = dict(os.environ)
    if cores():
        environment["TBAEM_CPU"] = str(cores()[worker % len(cores())])
    log_path = log_path or output / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    if result.returncode:
        raise RuntimeError(f"Worker exited {result.returncode}; inspect {log_path}. Resume retains started fits as interrupted.")


def smoke(output):
    import numpy as np
    from analyze import analyze
    source_hash = freeze(output)
    settings = protocol()
    representative = {c.label: c for c in cells(settings) if c.replicate == 0}
    for cell in representative.values():
        data = generate(cell, settings)
        assert data["holdout"].sum() == round(.1 * cell.n_subjects * cell.n_items)
        assert np.isfinite(data["train"]).any(axis=0).all()
        assert np.isfinite(data["train"]).any(axis=1).all()
    specifications = [("primary", "1pl", 0.), ("primary", "2pl", -.4),
                      ("primary", "2pl", 0.), ("primary", "2pl", .4),
                      ("contaminated_normal", "2pl", .4), ("clustered_items", "2pl", .4)]
    coordinates = []
    production_seeds = {c.seed for c in cells(settings)}
    for condition, model, rho in specifications:
        temp = Cell(condition, model, 12, 40, rho, 0, 0)
        seed = seed_for(settings, temp.label, 0, "smoke")
        assert seed not in production_seeds
        cell = Cell(**{**asdict(temp), "seed": seed})
        coordinates.append(cell)
        path = output / "smoke" / (cell.label + ".json")
        write_json(path, asdict(cell))
        print(f"Smoke: {cell.label} (full optimizer budgets)", flush=True)
        child(cell, output / "smoke", path)
    report = analyze(output / "smoke", coordinates, source_hash, "smoke")
    passed = report["status"] == "complete" and report["failed"] == report["interrupted"] == 0
    write_json(output / "smoke/gate.json", {"status": "passed" if passed else "failed",
               "source_manifest_sha256": source_hash, "full_size_generators_checked": 24,
               "fit_summary": report, "time": time.time()})
    if not passed:
        raise RuntimeError("Smoke failed; inspect its records. Do not launch production or delete failed fits.")
    print(json.dumps(report, indent=2))


def capacity(output):
    pending = [c for c in expected() if not done(output, c)]
    # Uncompressed numerical arrays plus 8 MiB per matrix for traces/metadata,
    # then 50% headroom. Compression usually reduces this; traces can vary.
    raw = sum(10*c.n_subjects*c.n_items + len(methods(c)) *
              (32*.1*c.n_subjects*c.n_items + 256*(c.n_subjects+c.n_items)) + 8*1024**2
              for c in pending)
    required = 1.5*raw + 6*1024**3
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    return {"remaining_matrices": len(pending), "planning_free_gib": required/1024**3,
            "available_gib": free/1024**3, "storage_ok": free >= required,
            "note": "Conservative planning allowance, not a guaranteed maximum. Keep additional headroom."}


def check_gate(output):
    source_hash = verify(output)
    path = output / "smoke/gate.json"
    gate = json.loads(path.read_text()) if path.exists() else {}
    if gate.get("status") != "passed" or gate.get("source_manifest_sha256") != source_hash:
        raise RuntimeError("A passing smoke test for this exact environment and source is required.")
    return source_hash


def check(output):
    check_gate(output)
    result = capacity(output)
    if not result["storage_ok"]:
        raise RuntimeError(f"Storage gate failed: {result}")
    return result


def run(output, max_matrices, workers=1):
    print(json.dumps(check(output), indent=2), flush=True)
    # Each matrix still runs in its own single-threaded process with its own seed,
    # so the worker count changes scheduling and wall-clock time, not estimates.
    # Matrices are handed out in the same replicate-major order as a single worker.
    with (output / "execution_log.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "workers": workers, "cpu_count": os.cpu_count(),
                                 "pinned_cores": len(cores()), "host": platform.node(),
                                 "max_matrices": max_matrices}) + "\n")
    pending = (cell for cell in expected() if not done(output, cell))
    guard, halt, errors = threading.Lock(), threading.Event(), []
    started = 0

    def claim():
        nonlocal started
        with guard:
            if halt.is_set() or (max_matrices and started >= max_matrices):
                return None
            if (output / "STOP").exists():
                print("Stopped between matrices. Remove STOP to resume.", flush=True)
                halt.set()
                return None
            if shutil.disk_usage(output).free < 6*1024**3:
                errors.append(RuntimeError("Less than 6 GiB remains. Stopping before the next matrix."))
                halt.set()
                return None
            cell = next(pending, None)
            started += cell is not None
            return cell

    def work(index):
        progress = output / ("progress.json" if workers == 1 else f"progress/worker_{index:03d}.json")
        log_path = output / ("run.log" if workers == 1 else f"logs/worker_{index:03d}.log")
        while (cell := claim()) is not None:
            try:
                verify(output)
                write_json(progress, {"cell": cell.label, "replicate": cell.replicate,
                           "time": time.time(), "pid": os.getpid(), "worker": index})
                print(f"Starting {cell.label}, replication {cell.replicate + 1}/2000", flush=True)
                child(cell, output, log_path=log_path, worker=index)
            except Exception as error:
                with guard:
                    errors.append(error)
                    halt.set()
                return

    threads = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    write_json(output / "status.json", status(output))
    print(f"Saved status to {output / 'status.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["smoke", "check", "run", "status", "analyze"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-matrices", type=int, default=0,
                        help="Stop after this many new matrices; 0 means all. Does not change the 2000 target.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Matrices fitted at once, each in its own single-threaded process. Use at most one per physical core.")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if args.max_matrices < 0:
        parser.error("--max-matrices must be nonnegative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        print(json.dumps(status(output), indent=2))
        return
    with exclusive_lock(output / "supervisor.lock"):
        if args.action == "smoke":
            smoke(output)
        elif args.action == "check":
            print(json.dumps(check(output), indent=2))
        elif args.action == "run":
            run(output, args.max_matrices, args.workers)
        else:
            from analyze import analyze
            result = analyze(output, expected(), verify(output), "production")
            print(json.dumps(result, indent=2))
            if args.require_complete and result["status"] != "complete":
                raise SystemExit("Not complete: these are interim summaries.")


if __name__ == "__main__":
    main()
