"""Design, seed, recovery and resume checks; no production fits."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from design import Cell, cells, generate, methods, latent_items
from protocol_tools import protocol
from recovery import identify, probability_invariance
from mcstats import summarize_errors
from launch import status


def test_design():
    coordinates = list(cells(protocol()))
    assert len(coordinates) == 48000
    assert len({c.label for c in coordinates}) == 24
    assert len({c.seed for c in coordinates}) == 48000
    assert sum(len(methods(c)) for c in coordinates) == 184000
    assert coordinates == list(cells(protocol()))


def test_manifest_covers_nested_estimator_sources():
    from protocol_tools import manifest
    assert "transposed_em/families/normal.py" in manifest()["sources"]


def test_frozen_protocol_rejects_changes(tmp_path, monkeypatch):
    import protocol_tools
    monkeypatch.setattr(protocol_tools, "manifest", lambda: {"version": 1})
    original = protocol_tools.freeze(tmp_path)
    assert protocol_tools.verify(tmp_path) == original
    monkeypatch.setattr(protocol_tools, "manifest", lambda: {"version": 2})
    with pytest.raises(RuntimeError, match="changed"):
        protocol_tools.verify(tmp_path)


def test_duplicate_worker_is_refused(tmp_path):
    from filelock import Timeout
    from run import exclusive_lock
    with exclusive_lock(tmp_path / "worker.lock"):
        with pytest.raises(Timeout):
            with exclusive_lock(tmp_path / "worker.lock"):
                pytest.fail("A second worker acquired the same lock")


@pytest.mark.parametrize("condition,model,rho", [
    ("primary", "1pl", 0.), ("primary", "2pl", -.4), ("primary", "2pl", 0.),
    ("primary", "2pl", .4), ("contaminated_normal", "2pl", .4), ("clustered_items", "2pl", .4)])
def test_generation(condition, model, rho):
    cell = Cell(condition, model, 12, 40, rho, 0, 123)
    first, second = generate(cell, protocol()), generate(cell, protocol())
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["holdout"].sum() == 48
    assert np.isfinite(first["train"]).any(axis=0).all()
    assert np.isfinite(first["train"]).any(axis=1).all()
    assert abs(np.log(first["a"]).mean()) < 1e-14
    transformed, _ = identify(first["theta"] + first["b"].mean(), first["b"], np.log(first["a"]), model)
    rows, cols = np.where(first["holdout"])
    assert probability_invariance(first["theta"] + first["b"].mean(), first["b"], np.log(first["a"]), transformed, rows, cols) < 1e-12


@pytest.mark.parametrize("condition", ["primary", "contaminated_normal", "clustered_items"])
def test_population_moments(condition):
    cell = Cell(condition, "2pl", 20, 200000, .4, 0, 17)
    latent, _ = latent_items(np.random.default_rng(17), cell, protocol())
    np.testing.assert_allclose(np.cov(latent.T), [[.64, .112], [.112, .1225]], atol=.02)


def test_mcse_uses_replications():
    error = np.array([[1., -1.], [3., -3.]])
    result = summarize_errors(error)
    assert result["bias"] == 0
    assert result["mse"] == 5
    assert result["mcse_mse"] == 4
    assert np.isnan(summarize_errors(error[:1])["mcse_bias"])


def test_interrupted_and_failed_are_not_replaced(tmp_path, monkeypatch):
    import run
    settings = protocol()
    cell = Cell("primary", "1pl", 12, 40, 0., 0, 999)
    def fail(*args, **kwargs):
        raise RuntimeError("intentional test failure")
    monkeypatch.setattr(run, "fit_one", fail)
    destination = tmp_path / "recovery" / cell.label / "rep_000"
    interrupted = destination / "tbaem_d"
    interrupted.mkdir(parents=True)
    (interrupted / "started.json").write_text("{}")
    run.run_cell(cell, settings, tmp_path, "test", "smoke")
    paths = list(destination.glob("*/summary.json"))
    before = {p: p.read_bytes() for p in paths}
    assert json.loads((interrupted / "summary.json").read_text())["status"] == "interrupted"
    run.run_cell(cell, settings, tmp_path, "test", "smoke")
    assert before == {p: p.read_bytes() for p in paths}


def test_analysis_audits_saved_vectors(tmp_path, monkeypatch):
    import run
    from analyze import analyze
    cell = Cell("primary", "2pl", 12, 40, .4, 0, 55)
    def known(data, model, method, *args):
        return {"theta": data["theta"] + data["b"].mean(), "b": data["b"],
                "log_a": np.log(data["a"]), "history": [],
                "diagnostics": {"stopping_reason": "iteration_limit"}}
    monkeypatch.setattr(run, "fit_one", known)
    run.run_cell(cell, protocol(), tmp_path, "test", "smoke")
    report = analyze(tmp_path, [cell], "test", "smoke")
    assert report["attempted_fits"] == report["returned"] == 4
    path = next(tmp_path.glob("recovery/*/rep_000/tbaem_d/summary.json"))
    value = json.loads(path.read_text())
    value["metrics"]["theta"]["mse"] = 100
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        analyze(tmp_path, [cell], "test", "smoke")


@pytest.mark.parametrize("workers", [1, 4])
def test_parallel_run_fits_each_matrix_once(tmp_path, monkeypatch, workers):
    import threading
    import launch
    coordinates = [Cell("primary", "1pl", 12, 40, 0., rep, 100 + rep) for rep in range(10)]
    calls, workers_used, guard = [], set(), threading.Lock()
    def fake_child(cell, output, smoke_coordinate=None, log_path=None, worker=0):
        with guard:
            calls.append(cell)
            workers_used.add(worker)
        for method in methods(cell):
            path = launch.directory(output, cell) / method / "summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"cell": asdict(cell), "method": method, "status": "returned"}))
    monkeypatch.setattr(launch, "expected", lambda: coordinates)
    monkeypatch.setattr(launch, "check", lambda output: {})
    monkeypatch.setattr(launch, "verify", lambda output: "test")
    monkeypatch.setattr(launch, "child", fake_child)
    launch.run(tmp_path, 3, workers)
    assert len(calls) == 3
    launch.run(tmp_path, 0, workers)
    assert sorted(calls, key=lambda c: c.replicate) == coordinates
    assert workers_used <= set(range(workers))
    assert json.loads((tmp_path / "execution_log.jsonl").read_text().splitlines()[-1])["workers"] == workers
    (tmp_path / "STOP").touch()
    coordinates.append(Cell("primary", "1pl", 12, 40, 0., 10, 110))
    launch.run(tmp_path, 0, workers)
    assert len(calls) == 10
