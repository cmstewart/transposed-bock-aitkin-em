"""Replication-level Monte Carlo errors, separate from parameter uncertainty."""
import numpy as np

STABLE = {"fixed_point_criteria", "gradient_and_objective_criteria",
          "training_elbo_and_parameter_stability"}


def mcse(values):
    values = np.asarray(values, float)
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def summarize_errors(error):
    error = np.asarray(error, float)
    if error.ndim == 1:
        error = error[:, None]
    bias, mae, mse = error.mean(axis=1), np.abs(error).mean(axis=1), (error**2).mean(axis=1)
    rmse = float(np.sqrt(mse.mean()))
    return {"replicates": len(error), "parameters_per_replicate": error.shape[1],
            "bias": float(bias.mean()), "mcse_bias": mcse(bias),
            "mae": float(mae.mean()), "mcse_mae": mcse(mae),
            "mse": float(mse.mean()), "mcse_mse": mcse(mse),
            "rmse": rmse, "mcse_rmse": mcse(mse) / (2 * rmse) if rmse > 0 else mcse(mse),
            "mean_rep_rmse": float(np.sqrt(mse).mean()), "mcse_mean_rep_rmse": mcse(np.sqrt(mse))}
