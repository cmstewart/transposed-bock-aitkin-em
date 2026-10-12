"""Audited adapters for the installed torch_measure and py-irt models."""
from __future__ import annotations

import time

import numpy as np
from scipy.special import expit, logit


def starting_values(train):
    observed = np.isfinite(train)
    y = np.nan_to_num(train)
    row_p = (y.sum(axis=1) + 0.5) / (observed.sum(axis=1) + 1)
    col_p = (y.sum(axis=0) + 0.5) / (observed.sum(axis=0) + 1)
    theta = logit(row_p)
    b = theta.mean() - logit(col_p)
    return theta - b.mean(), b - b.mean()


def fit_jml(train, model, seed, max_epochs=300, legacy=False):
    import torch
    from torch_measure.models.rasch import Rasch
    from torch_measure.models.twopl import TwoPL
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    fitted = (Rasch if model == "1pl" else TwoPL)(*train.shape, device="cpu")
    rows, cols = np.where(np.isfinite(train))
    query = {"subject_idx": torch.as_tensor(rows), "item_idx": torch.as_tensor(cols)}
    started = time.perf_counter()
    if legacy:
        history = fitted.fit(torch.tensor(train, dtype=torch.float32), method="jml",
                             max_epochs=max_epochs, lr=.1, regularization=.01,
                             convergence_tol=1e-6, verbose=False)
        trace = [{"iteration": i + 1, "objective": float(x)} for i, x in enumerate(history["losses"])]
        reason = "legacy_loss_change" if len(trace) < max_epochs else "iteration_limit"
        diagnostics = {}
    else:
        fitted = fitted.double()
        theta, b = starting_values(train)
        with torch.no_grad():
            fitted.ability.copy_(torch.tensor(theta))
            fitted.difficulty.copy_(torch.tensor(b))
            if model == "2pl":
                fitted._discrimination_raw.zero_()
        y = torch.tensor(train[rows, cols], dtype=torch.float64)
        parameters = list(fitted.parameters())
        count = sum(p.numel() for p in parameters)
        optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=20, history_size=20,
                        tolerance_grad=1e-7, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            p = fitted.predict(query).clamp(1e-7, 1 - 1e-7)
            nll = -(y * p.log() + (1 - y) * torch.log1p(-p)).mean()
            penalty = .01 * sum(x.square().sum() for x in parameters) / count
            loss = train.shape[1] * (nll + penalty)
            loss.backward()
            return loss

        trace, stable = [], 0
        reason = "iteration_limit"
        for epoch in range(1, max_epochs + 1):
            optimizer.step(closure)
            objective = float(closure().detach())
            gradient = max(float(p.grad.abs().max()) for p in parameters)
            relative = np.inf if not trace else abs(objective - trace[-1]["objective"]) / (1 + abs(objective))
            stable = stable + 1 if epoch >= 10 and gradient < 1e-5 and relative < 1e-9 else 0
            trace.append({"iteration": epoch, "objective": objective,
                          "scaled_gradient_max": gradient, "relative_change": relative,
                          "stable_iterations": stable})
            if stable >= 5:
                reason = "gradient_and_objective_criteria"
                break
        diagnostics = {"scaled_gradient_max": gradient,
                       "objective_scale": train.shape[1], "regularization": .01}
    with torch.no_grad():
        theta = fitted.ability.cpu().numpy().astype(float)
        b = fitted.difficulty.cpu().numpy().astype(float)
        log_a = fitted._discrimination_raw.cpu().numpy().astype(float) if model == "2pl" else np.zeros_like(b)
        probe = np.linspace(0, len(rows) - 1, min(4096, len(rows)), dtype=int)
        native = fitted.predict({key: value[probe] for key, value in query.items()}).cpu().numpy()
    formula = expit(np.exp(log_a[cols[probe]]) * (theta[rows[probe]] - b[cols[probe]]))
    diagnostics.update(native_export_probability_error=float(np.max(np.abs(native - formula))),
                       elapsed_seconds=time.perf_counter() - started, stopping_reason=reason,
                       iterations=len(trace))
    return {"theta": theta, "b": b, "log_a": log_a, "history": trace,
            "diagnostics": diagnostics}


def deterministic_vi_objective(state, rows, cols, y, n_observed, n_subjects, n_items, model, order=9, prior_policy="native"):
    """Fixed training-probe ELBO with analytic KL; used only as a stopping diagnostic."""
    from numpy.polynomial.hermite import hermgauss
    nodes, weights = hermgauss(order)
    weights = weights / np.sqrt(np.pi)
    mean = state["loc_ability"][rows] - state["loc_diff"][cols]
    sd = np.sqrt(state["scale_ability"][rows]**2 + state["scale_diff"][cols]**2)
    difference = mean[:, None] + np.sqrt(2.) * sd[:, None] * nodes
    if model == "1pl":
        linear = difference
        likelihood = np.sum(weights * (y[:, None] * linear - np.logaddexp(0., linear)), axis=1)
    else:
        alpha = state["loc_slope"][cols, None] + np.sqrt(2.) * state["scale_slope"][cols, None] * nodes
        linear = difference[:, :, None] * np.exp(alpha[:, None, :])
        likelihood = np.einsum("i,j,kij->k", weights, weights,
                               y[:, None, None] * linear - np.logaddexp(0., linear))
    # These are the vague prior scales in py-irt 0.7.1, not new fitted priors.
    diff_sd = (1.e3 if model == "1pl" else .1) if prior_policy == "native" else 1.5
    slope_sd = .1 if prior_policy == "native" else .5
    priors = [("ability", 1.), ("diff", diff_sd)]
    if model == "2pl":
        priors.append(("slope", slope_sd))
    kl = 0.
    for key, prior_sd in priors:
        mean, sd = state["loc_" + key], state["scale_" + key]
        kl += np.sum(np.log(prior_sd / sd) + (sd**2 + mean**2) / (2 * prior_sd**2) - .5)
    return float(-likelihood.mean() + kl / n_observed)


def fit_vi(train, model, seed, max_epochs=4000, legacy=False, prior_policy="regular"):
    import pyro
    import torch
    from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
    from py_irt.models import IrtModel
    torch.set_num_threads(1)
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    rows, cols = np.where(np.isfinite(train))
    subjects, items = torch.as_tensor(rows), torch.as_tensor(cols)
    responses = torch.as_tensor(train[rows, cols], dtype=torch.float32)
    fitted = IrtModel.from_name(model)(device="cpu", num_items=train.shape[1],
               num_subjects=train.shape[0], priors="vague", verbose=False)
    guide, model_fn = fitted.get_guide(), fitted.get_model()
    prior_policy = "native" if legacy else prior_policy
    if prior_policy not in {"native", "regular"}:
        raise ValueError("Unknown prior policy")
    if prior_policy == "regular":
        from pyro.poutine.messenger import Messenger
        import pyro.distributions as dist

        class ItemPriors(Messenger):
            def _pyro_sample(self, message):
                original = message["fn"]
                if message["name"] == "b":
                    message["fn"] = dist.Normal(original.loc, torch.full_like(original.scale, 1.5))
                elif message["name"] == "a":
                    base = original.base_dist
                    message["fn"] = dist.LogNormal(base.loc, torch.full_like(base.scale, .5))

        model_fn = ItemPriors()(model_fn)
    guide(subjects, items, responses)
    if not legacy:
        theta, b = starting_values(train)
        values = {"loc_ability": theta, "loc_diff": b,
                  "scale_ability": np.full(len(theta), .1), "scale_diff": np.full(len(b), .3)}
        if model == "2pl":
            values.update(loc_slope=np.zeros(len(b)), scale_slope=np.full(len(b), .1))
        with torch.no_grad():
            for key, value in values.items():
                parameter = pyro.param(key)
                target = torch.tensor(np.log(value) if key.startswith("scale") else value, dtype=parameter.dtype)
                parameter.unconstrained().copy_(target)
    scheduler = pyro.optim.ExponentialLR({"optimizer": torch.optim.Adam,
                 "optim_args": {"lr": .05 if legacy else .03},
                 "gamma": .995 if legacy else .003 ** (1 / max_epochs)})
    svi = SVI(model_fn, guide, scheduler, loss=Trace_ELBO() if legacy else TraceMeanField_ELBO())
    probe = np.linspace(0, len(rows) - 1, min(2048, len(rows)), dtype=int)
    started = time.perf_counter()
    trace, losses, previous_state = [], [], None
    stable, reason = 0, "iteration_limit"
    best_state, best_loss = None, np.inf

    def state():
        return {key: value.detach().cpu().numpy().copy() for key, value in pyro.get_param_store().items()}

    for epoch in range(1, max_epochs + 1):
        losses.append(float(svi.step(subjects, items, responses)) / len(rows))
        scheduler.step()
        window = 25 if legacy else 100
        if epoch % window:
            continue
        current = state()
        objective = float(np.mean(losses[-window:])) if legacy else deterministic_vi_objective(
            current, rows[probe], cols[probe], train[rows[probe], cols[probe]], len(rows), *train.shape, model,
            prior_policy=prior_policy)
        relative = np.inf if not trace else abs(objective - trace[-1]["objective"]) / max(1., abs(objective))
        parameter_change = np.inf if previous_state is None else max(
            float(np.sqrt(np.mean((current[key] - previous_state[key])**2))) for key in current)
        if legacy and objective < best_loss:
            best_loss, best_state = objective, {key: value.copy() for key, value in current.items()}
        criteria = (epoch >= (200 if legacy else 1000) and relative < (1e-3 if legacy else 2e-4)
                    and (legacy or parameter_change < 2e-3))
        stable = stable + 1 if criteria else 0
        trace.append({"iteration": epoch, "objective": objective, "relative_change": relative,
                      "parameter_rms_change": parameter_change, "stable_windows": stable,
                      "mean_stochastic_loss": float(np.mean(losses[-window:]))})
        previous_state = current
        if stable >= 4:
            reason = "legacy_smoothed_elbo" if legacy else "training_elbo_and_parameter_stability"
            break
    final = best_state if legacy and best_state is not None else state()
    if legacy and best_state is not None:
        with torch.no_grad():
            for key, value in final.items():
                parameter = pyro.param(key)
                parameter.unconstrained().copy_(torch.tensor(np.log(value) if key.startswith("scale") else value))
    exported = fitted.export()
    theta, b = np.asarray(exported["ability"]), np.asarray(exported["diff"])
    log_a = np.log(np.asarray(exported["disc"])) if model == "2pl" else np.zeros_like(b)
    export_error = max(float(np.max(np.abs(theta - final["loc_ability"]))),
                       float(np.max(np.abs(b - final["loc_diff"]))))
    if model == "2pl":
        export_error = max(export_error, float(np.max(np.abs(log_a - final["loc_slope"]))))
    return {"theta": theta, "b": b, "log_a": log_a, "history": trace, "guide": final,
            "diagnostics": {"stopping_reason": reason, "iterations": epoch,
                            "elapsed_seconds": time.perf_counter() - started,
                            "native_export_parameter_error": export_error,
                            "initial_scale_diff": (1000. if model == "1pl" else 10.) if legacy else .3,
                            "prior_policy": prior_policy,
                            "prior_sd_diff": (1000. if model == "1pl" else .1) if prior_policy == "native" else 1.5,
                            "prior_sd_log_a": None if model == "1pl" else (.1 if prior_policy == "native" else .5),
                            "prediction_type": "plug_in_guide_locations_and_median_discrimination",
                            "parameter_inference_validated": False}}
