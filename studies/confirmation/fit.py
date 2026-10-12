"""Shared fitting adapters for simulated or empirical binary response matrices."""
import numpy as np

from transposed_em import LaplaceIRT
from baselines import fit_jml, fit_vi
from recovery import population
from protocol_tools import write_json


def fit_one(data, model, method, seed, settings, progress_path=None):
    if method in {"tbaem_d", "tbaem_c"}:
        callback = None if progress_path is None else lambda row: write_json(progress_path, row)
        fit = LaplaceIRT(model=model,
                         covariance="diagonal" if method == "tbaem_d" else "full",
                         progress_callback=callback, **settings["transposed"]).fit(data["train"])
        nodes, weights = fit.posterior_nodes_, fit.node_weights_
        b = np.einsum("jk,jk->j", weights, nodes[:, :, 0])
        log_a = (np.einsum("jk,jk->j", weights, nodes[:, :, 1])
                 if model == "2pl" else np.zeros_like(b))
        rows, cols = np.where(data["holdout"])
        return {"theta": fit.theta_.copy(), "b": b, "log_a": log_a,
                "probability": fit.predict_pairs(rows, cols),
                "history": fit.history_, "diagnostics": fit.diagnostics_,
                "population": population(fit.family_moments_.raw_cov),
                "item_point_estimate": "E(b), E(log_a), exp(E(log_a)); not E(a)",
                "posterior_covariance": fit.item_posterior_covariance_,
                "posterior_modes": fit.latent_modes_}
    if method == "jml":
        return fit_jml(data["train"], model, seed, **settings["jml_runtime"])
    if method == "vi":
        return fit_vi(data["train"], model, seed, **settings["vi_runtime"])
    raise ValueError(f"Unknown method: {method}")
