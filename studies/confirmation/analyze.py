"""Audit saved vectors and report each parameter family without pooling studies."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import itertools
import json
from pathlib import Path

from design import Cell, cells, methods
from protocol_tools import OUTPUT, protocol, verify
from run import write_json
from mcstats import mcse, summarize_errors, STABLE
import numpy as np
import pandas as pd


def audit_records(output, expected, source_hash, phase):
    records = []
    expected_map = {(c.label, c.replicate, m): c for c in expected for m in methods(c)}
    for path in sorted((output / "recovery").glob("*/rep_*/*/summary.json")):
        value = json.loads(path.read_text())
        key = (path.parents[2].name, int(path.parents[1].name.removeprefix("rep_")), path.parent.name)
        if key not in expected_map or value["cell"] != asdict(expected_map[key]):
            raise ValueError(f"Unexpected seed or condition: {path}")
        if value["source_manifest_sha256"] != source_hash or value["phase"] != phase or value["method"] != key[2]:
            raise ValueError(f"Wrong protocol, phase or method: {path}")
        if value["status"] not in {"returned", "failed", "interrupted"}:
            raise ValueError(f"Unknown outcome: {path}")
        if value["status"] == "returned":
            for name, metric_key in (("parameters", "metrics"), ("population", "population_metrics")):
                if metric_key not in value:
                    continue
                with np.load(path.parent / f"{name}.npz", allow_pickle=False) as archive:
                    for parameter, saved in value[metric_key].items():
                        estimate, truth, error = [archive[f"{parameter}_{kind}"] for kind in ("estimate", "truth", "error")]
                        assert estimate.shape == truth.shape == error.shape
                        assert all(np.isfinite(v).all() for v in (estimate, truth, error))
                        np.testing.assert_array_equal(estimate - truth, error)
                        for statistic, actual in (("bias", error.mean()), ("mae", np.abs(error).mean()),
                                                  ("mse", (error**2).mean()), ("rmse", np.sqrt((error**2).mean()))):
                            np.testing.assert_allclose(saved[statistic], actual, rtol=1e-12, atol=1e-14)
            with np.load(path.parent / "predictions.npz", allow_pickle=False) as prediction:
                p, y = prediction["probability"], prediction["y"]
                assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                np.testing.assert_allclose(value["brier"], ((p - y)**2).mean(), atol=1e-14)
                clipped = np.clip(p, 1e-12, 1 - 1e-12)
                np.testing.assert_allclose(value["logloss"], -np.mean(y * np.log(clipped) + (1-y) * np.log1p(-clipped)), atol=1e-14)
        value["path"] = path
        value["stable"] = value.get("diagnostics", {}).get("stopping_reason") in STABLE
        records.append(value)
    return records


def analyze(output, expected, source_hash, phase):
    records = audit_records(output, expected, source_hash, phase)
    destination = output / "analysis"
    destination.mkdir(exist_ok=True)
    attempts = []
    for record in records:
        diagnostics = record.get("diagnostics", {})
        attempts.append({**record["cell"], "cell_label": record["cell_label"], "method":record["method"],
            "status":record["status"], "rule_stable":record["stable"], "reason":record.get("reason"),
            "stopping_reason":diagnostics.get("stopping_reason"),
            "iterations":diagnostics.get("iterations",diagnostics.get("iteration")),
            "theta_score_max":diagnostics.get("theta_score_max"),
            "covariance_score_max":diagnostics.get("covariance_score_max"),
            "elapsed_seconds":record.get("elapsed_seconds"), "brier":record.get("brier"),
            "logloss":record.get("logloss")})
    pd.DataFrame(attempts).to_csv(destination/"attempts.csv",index=False)
    groups = defaultdict(list)
    for r in records:
        groups[(r["cell_label"], r["method"])].append(r)
    planned = defaultdict(int)
    for cell in expected:
        for method in methods(cell):
            planned[(cell.label, method)] += 1
    coverage, performance, recovery, pop_rows, truth_bins = [], [], [], [], []
    bins = {"theta": [-np.inf, -1.5, -.5, .5, 1.5, np.inf],
            "b": [-np.inf, -1., -.5, 0., .5, 1., np.inf],
            "log_a": [-np.inf, -.5, -.25, 0., .25, .5, np.inf],
            "a": [0., .5, .8, 1., 1.25, 2., np.inf]}
    for (label, method), target in sorted(planned.items()):
        group = groups[(label, method)]
        returned = [r for r in group if r["status"] == "returned"]
        coverage.append(dict(cell=label, method=method, planned=target, attempted=len(group),
            returned=len(returned), rule_stable=sum(r["stable"] for r in returned),
            limits=sum(r.get("diagnostics", {}).get("stopping_reason") == "iteration_limit" for r in returned),
            other_numerical_stops=sum(not r["stable"] and r.get("diagnostics", {}).get("stopping_reason") != "iteration_limit" for r in returned),
            failed=sum(r["status"] == "failed" for r in group),
            interrupted=sum(r["status"] == "interrupted" for r in group)))
        for outcome in ("returned", "rule_stable", "limits", "other_numerical_stops", "failed", "interrupted"):
            rate = coverage[-1][outcome] / len(group) if group else np.nan
            coverage[-1][outcome + "_rate"] = rate
            coverage[-1][outcome + "_mcse"] = np.sqrt(rate * (1-rate) / len(group)) if group else np.nan
        for cohort in ("all_returned", "rule_stable"):
            selected = returned if cohort == "all_returned" else [r for r in returned if r["stable"]]
            if not selected:
                continue
            meta = dict(cell=label, method=method, cohort=cohort, planned=target, attempted=len(group),
                        condition=selected[0]["cell"]["condition"], model=selected[0]["cell"]["model"])
            for metric in ("brier", "logloss", "elapsed_seconds"):
                values = np.array([r[metric] for r in selected])
                performance.append({**meta, "metric": metric, "replicates": len(values),
                                    "mean": values.mean(), "mcse": mcse(values)})
            for archive_name, key, container in (("parameters", "metrics", recovery),
                                                  ("population", "population_metrics", pop_rows)):
                if key not in selected[0]:
                    continue
                for parameter in selected[0][key]:
                    errors, truths = [], []
                    for r in selected:
                        with np.load(r["path"].parent / f"{archive_name}.npz", allow_pickle=False) as archive:
                            errors.append(archive[parameter + "_error"])
                            truths.append(archive[parameter + "_truth"])
                    errors, truths = np.stack(errors), np.stack(truths)
                    role = "fixed" if ((meta["model"] == "1pl" and parameter in {"a", "log_a"}) or
                                      (method == "tbaem_d" and parameter in {"cov_b_log_a", "corr_b_log_a"})) else "estimated"
                    target_description = selected[0].get("population_reference") if archive_name == "population" else "realized_generating_parameter"
                    correlations = [r[key][parameter]["spearman"] for r in selected if "spearman" in r[key][parameter]]
                    container.append({**meta, "parameter": parameter, "role": role, "target": target_description,
                        **summarize_errors(errors), "mean_spearman": np.mean(correlations) if correlations else np.nan,
                        "mcse_spearman": mcse(correlations)})
                    if archive_name != "parameters" or role == "fixed":
                        continue
                    for lower, upper in zip(bins[parameter][:-1], bins[parameter][1:]):
                        within = (truths >= lower) & (truths < upper)
                        active = within.any(axis=1)
                        if not active.any():
                            continue
                        per_bias = np.array([e[w].mean() for e,w in zip(errors[active], within[active])])
                        per_mse = np.array([(e[w]**2).mean() for e,w in zip(errors[active], within[active])])
                        rmse = np.sqrt(per_mse.mean())
                        truth_bins.append({**meta, "parameter": parameter, "lower": lower, "upper": upper,
                            "replicates": int(active.sum()), "parameters": int(within.sum()),
                            "bias": per_bias.mean(), "mcse_bias": mcse(per_bias), "rmse": rmse,
                            "mcse_rmse": mcse(per_mse)/(2*rmse) if rmse > 0 else 0.})
    paired = []
    labels = sorted({c.label for c in expected})
    for label in labels:
        candidates = sorted({m for l,m in planned if l == label})
        for left, right in itertools.combinations(candidates, 2):
            for cohort in ("all_returned", "rule_stable"):
                def eligible(method):
                    return {r["cell"]["replicate"]: r for r in groups[(label, method)] if r["status"] == "returned"
                            and (cohort == "all_returned" or r["stable"])}
                a, b = eligible(left), eligible(right)
                reps = sorted(a.keys() & b.keys())
                if not reps:
                    continue
                for metric in ("brier", "logloss", "theta_rmse", "b_rmse", "log_a_rmse", "a_rmse"):
                    if metric.endswith("_rmse"):
                        parameter = metric.removesuffix("_rmse")
                        if a[reps[0]]["cell"]["model"] == "1pl" and parameter in {"a", "log_a"}:
                            continue
                        mse_a = np.array([a[r]["metrics"][parameter]["mse"] for r in reps])
                        mse_b = np.array([b[r]["metrics"][parameter]["mse"] for r in reps])
                        ra, rb = np.sqrt(mse_a.mean()), np.sqrt(mse_b.mean())
                        difference = ra - rb
                        error = mcse(mse_a/(2*ra) - mse_b/(2*rb)) if ra > 0 and rb > 0 else np.nan
                    else:
                        delta = np.array([a[r][metric] - b[r][metric] for r in reps])
                        difference, error = delta.mean(), mcse(delta)
                    paired.append(dict(cell=label, left=left, right=right, cohort=cohort, metric=metric,
                                       paired_replicates=len(reps), left_minus_right=difference, mcse=error))
    for name, rows in (("completion_by_cell_method",coverage), ("performance",performance),
                       ("parameter_recovery",recovery), ("population_recovery",pop_rows),
                       ("truth_bin_recovery",truth_bins), ("paired_contrasts",paired)):
        pd.DataFrame(rows).to_csv(destination / f"{name}.csv", index=False)
    complete = all(r["attempted"] == r["planned"] for r in coverage)
    summary = {"status": "complete" if complete else "incomplete", "phase": phase,
               "attempted_fits": len(records), "planned_fits": sum(planned.values()),
               "returned": sum(r["status"] == "returned" for r in records),
               "rule_stable": sum(r["status"] == "returned" and r["stable"] for r in records),
               "failed": sum(r["status"] == "failed" for r in records),
               "interrupted": sum(r["status"] == "interrupted" for r in records),
               "conditions": len(labels), "source_manifest_sha256": source_hash}
    write_json(destination / "completion_manifest.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = analyze(args.output, list(cells(protocol())), verify(args.output), "production")
    print(json.dumps(result, indent=2))
    if args.require_complete and result["status"] != "complete":
        raise SystemExit("Study is incomplete; no final manuscript tables are produced")
