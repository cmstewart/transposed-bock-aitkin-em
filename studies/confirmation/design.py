"""Frozen condition coordinates and response generation for the confirmation study."""
from dataclasses import asdict, dataclass
import hashlib

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class Cell:
    condition: str
    model: str
    n_subjects: int
    n_items: int
    item_correlation: float
    replicate: int
    seed: int

    @property
    def label(self):
        return (f"{self.condition}_{self.model}_n{self.n_subjects}_p{self.n_items}"
                f"_rho{self.item_correlation:+.1f}")


def seed_for(protocol, label, replicate, phase="production"):
    value = f"{protocol['base_seed']}:{phase}:{label}:{replicate}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little") % (2**31 - 1)


def cells(protocol, phase="production"):
    used_seeds = set()
    for condition in protocol["conditions"]:
        for model in condition["models"]:
            for n in protocol["n_subjects"]:
                for p in protocol["n_items"]:
                    correlations = [0.] if model == "1pl" else condition["correlations_2pl"]
                    for rho in correlations:
                        for rep in range(protocol["repetitions"]):
                            cell = Cell(condition["name"], model, n, p, rho, rep, 0)
                            seed = seed_for(protocol, cell.label, rep, phase)
                            # Resolve 31-bit hash collisions before any data are generated.
                            while seed in used_seeds:
                                seed = (seed + 1) % (2**31 - 1)
                            used_seeds.add(seed)
                            yield Cell(**{**asdict(cell), "seed": seed})


def methods(cell):
    return ["tbaem_d", "jml", "vi"] if cell.model == "1pl" else ["tbaem_d", "tbaem_c", "jml", "vi"]


def latent_items(rng, cell, protocol):
    covariance = np.array([[.8**2, .8 * .35 * cell.item_correlation],
                           [.8 * .35 * cell.item_correlation, .35**2]])
    settings = next(x for x in protocol["conditions"] if x["name"] == cell.condition)
    cluster = np.arange(cell.n_items, dtype=np.int32)
    if cell.condition == "clustered_items":
        cluster //= settings["cluster_size"]
        shared = rng.multivariate_normal(np.zeros(2), covariance, int(cluster.max()) + 1)
        individual = rng.multivariate_normal(np.zeros(2), covariance, cell.n_items)
        fraction = settings["shared_variance_fraction"]
        latent = np.sqrt(fraction) * shared[cluster] + np.sqrt(1 - fraction) * individual
    else:
        latent = rng.multivariate_normal(np.zeros(2), covariance, cell.n_items)
        if cell.condition == "contaminated_normal":
            rate, multiplier = settings["contamination_rate"], settings["scale_multiplier"]
            scale = np.where(rng.random(cell.n_items) < rate, multiplier, 1.)
            scale /= np.sqrt((1 - rate) + rate * multiplier**2)
            latent *= scale[:, None]
        elif cell.condition != "primary":
            raise ValueError(f"Unknown generating condition: {cell.condition}")
    return latent, cluster


def split_holdout(rng, shape, rate):
    n, p = shape
    target = max(1, int(round(rate * n * p)))
    rows_left, cols_left = np.full(n, p), np.full(p, n)
    holdout = np.zeros(shape, dtype=bool)
    count = 0
    # A scalar counter avoids rescanning the full matrix for every held-out cell.
    for flat in rng.permutation(n * p):
        i, j = divmod(int(flat), p)
        if rows_left[i] > 1 and cols_left[j] > 1:
            holdout[i, j] = True
            rows_left[i] -= 1
            cols_left[j] -= 1
            count += 1
            if count == target:
                return holdout
    raise ValueError("Holdout target cannot preserve a training observation in every row and column")


def generate(cell, protocol):
    rng = np.random.default_rng(cell.seed)
    theta = rng.normal(size=cell.n_subjects)
    latent, cluster = latent_items(rng, cell, protocol)
    b, alpha = latent.T.copy()
    alpha = alpha - alpha.mean() if cell.model == "2pl" else np.zeros(cell.n_items)
    a = np.exp(alpha)
    probability = expit(a[None, :] * (theta[:, None] - b[None, :]))
    response = rng.binomial(1, probability).astype(np.uint8)
    holdout = split_holdout(rng, response.shape, protocol["holdout_rate"])
    train = response.astype(float)
    train[holdout] = np.nan
    return {"train": train, "response": response, "holdout": holdout,
            "theta": theta - b.mean(), "b": b, "a": a,
            "cluster_id": cluster, "generating_probability": probability}


def population_truth(cell):
    target = {"sd_b": .8, "variance_b": .64}
    if cell.model == "2pl":
        target.update(sd_log_a=.35, variance_log_a=.1225,
                      cov_b_log_a=.28 * cell.item_correlation,
                      corr_b_log_a=cell.item_correlation)
    return target
