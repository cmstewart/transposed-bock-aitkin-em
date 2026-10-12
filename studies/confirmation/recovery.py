"""Identified parameter exports and complete, non-overwriting recovery records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr


def identify(theta, b, log_a, model):
    theta, b, log_a = (np.asarray(x, dtype=float) for x in (theta, b, log_a))
    if theta.ndim != 1 or b.ndim != 1 or log_a.shape != b.shape:
        raise ValueError("Expected subject and item vectors")
    if not all(np.isfinite(x).all() for x in (theta, b, log_a)):
        raise ValueError("Nonfinite parameter export")
    shift = float(b.mean())
    log_scale = float(log_a.mean()) if model == "2pl" else 0.0
    scale = float(np.exp(log_scale))
    result = {"theta": scale * (theta - shift), "b": scale * (b - shift),
              "log_a": log_a - log_scale if model == "2pl" else np.zeros_like(b)}
    result["a"] = np.exp(result["log_a"])
    return result, {"shift": shift, "scale": scale}


def probability_invariance(theta, b, log_a, transformed, rows, cols):
    native = expit(np.exp(log_a[cols]) * (theta[rows] - b[cols]))
    exported = expit(transformed["a"][cols] *
                     (transformed["theta"][rows] - transformed["b"][cols]))
    return float(np.max(np.abs(native - exported)))


def parameter_metrics(estimate, truth):
    estimate, truth = np.asarray(estimate, float), np.asarray(truth, float)
    if estimate.shape != truth.shape or not estimate.size:
        raise ValueError("Estimate and truth must have matching nonempty shapes")
    if not (np.isfinite(estimate).all() and np.isfinite(truth).all()):
        raise ValueError("Nonfinite recovery vectors")
    error = estimate - truth
    result = {"count": int(error.size), "bias": float(error.mean()),
              "mae": float(np.abs(error).mean()), "mse": float(np.mean(error**2)),
              "rmse": float(np.sqrt(np.mean(error**2)))}
    if error.size > 1 and np.ptp(estimate) > 0 and np.ptp(truth) > 0:
        result["spearman"] = float(spearmanr(estimate, truth).statistic)
    return result


def save_recovery(path, estimates, truths, metadata):
    """Preserve every vector, its target, and a separate metric for each parameter."""
    arrays, metrics = {}, {}
    if set(estimates) != set(truths):
        raise ValueError("Each estimate needs an explicit target")
    for key in estimates:
        if not key.isidentifier():
            raise ValueError("Invalid parameter name")
        metrics[key] = parameter_metrics(estimates[key], truths[key])
        arrays[key + "_estimate"] = np.asarray(estimates[key])
        arrays[key + "_truth"] = np.asarray(truths[key])
        arrays[key + "_error"] = arrays[key + "_estimate"] - arrays[key + "_truth"]
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    return metrics


def population(covariance):
    cov = np.asarray(covariance)
    result = {"sd_b": np.sqrt(cov[0, 0]), "variance_b": cov[0, 0]}
    if cov.shape == (2, 2):
        result.update(sd_log_a=np.sqrt(cov[1, 1]), variance_log_a=cov[1, 1],
                      cov_b_log_a=cov[0, 1],
                      corr_b_log_a=cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    return {key: float(value) for key, value in result.items()}
