from __future__ import annotations

from typing import Dict, Tuple, List, Any
import numpy as np

def make_bounded_regression(
    n_samples: int = 512,
    n_features: int = 128,
    rho: float = 0.5,
    n_informative: int = 5,
    noise_std: float = 0.5,
    beta_k: float = 50.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Generate a synthetic bounded regression dataset with y in [0, 1].

    Returns:
      X: (n_samples, n_features)
      y: (n_samples,) in [0,1]
      info: dict with latent f, weights, and diagnostics
    """
    rng = np.random.default_rng(seed)

    # Correlated Gaussian features X ~ N(0, Sigma) with Toeplitz structure
    idx = np.arange(n_features)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n_samples, n_features)) @ L.T

    # Sparse informative weights for linear part
    w = np.zeros(n_features)
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)
    w[informative_idx] = rng.normal(0, 1.0, size=n_informative)

    # Latent signal: linear + mild nonlinear interactions
    f = X @ w
    if n_features >= 5:
        f += 0.5 * np.sin(X[:, 0]) + 0.3 * X[:, 1] * X[:, 2] + 0.5 * np.tanh(
            X[:, 3] - 0.5 * X[:, 4]
        )

    # Add Gaussian noise in latent (unbounded) space
    g = f + rng.normal(0, noise_std, size=n_samples)

    # Map to [0,1] via sigmoid
    p = 1.0 / (1.0 + np.exp(-g))
    p = np.clip(p, 1e-6, 1 - 1e-6)

    # Optional: add Beta noise on probability scale for continuous y in [0,1]
    alpha = beta_k * p
    beta = beta_k * (1 - p)
    y = rng.beta(alpha, beta)

    # Diagnostics
    corr_f_y = float(np.corrcoef(f, y)[0, 1])
    info = {
        "Sigma": Sigma,
        "weights": w,
        "informative_idx": informative_idx,
        "latent_f": f,
        "latent_g": g,
        "p": p,
        "corr_f_y": corr_f_y,
    }
    return X, y, info


def make_california_prob_dataset(
    test_size: float = 0.2,
    subsample_train: int = 4000,
    random_state: int = 42,
    subset_test: int = 200,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Load the California Housing dataset and map targets into (0,1) using a sigmoid
    after standardizing with training statistics. Optionally subsample the training set.

    Returns:
      X_train: (n_sub, d) training features (optionally standardized)
      y_train: (n_sub,) training targets in (0,1)
      test_sets: list of dicts with keys {name, X, y, meta}
    """
    # Local imports to avoid hard dependency at package import time
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Subsample training set
    rng = np.random.default_rng(random_state)
    n_sub = min(int(subsample_train), len(X_train))
    sel = rng.choice(len(X_train), size=n_sub, replace=False)
    X_train_raw_sub = X_train[sel]
    y_train_raw_sub = y_train_raw[sel]

    # Feature scaling
    if scale_features:
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train_raw_sub)
        X_test_proc = scaler.transform(X_test)
    else:
        X_train_proc = X_train_raw_sub
        X_test_proc = X_test

    # Target transform: standardize then sigmoid => (0,1)
    def _sigmoid(a: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-a))

    y_mean = float(y_train_raw_sub.mean())
    y_std = float(y_train_raw_sub.std() + 1e-12)

    def to_prob(a: np.ndarray) -> np.ndarray:
        z = (a - y_mean) / y_std
        return _sigmoid(z)

    y_train_proc = to_prob(y_train_raw_sub)
    y_test_proc = to_prob(y_test_raw)

    # Build test sets
    test_sets: List[Dict[str, Any]] = [
        {"name": "test_full", "X": X_test_proc, "y": y_test_proc, "meta": {"ids": list(range(len(y_test_proc)))}}
    ]
    if subset_test and len(X_test_proc) > subset_test:
        rng2 = np.random.default_rng(random_state + 7)
        sel_t = rng2.choice(len(X_test_proc), size=subset_test, replace=False)
        test_sets.append(
            {
                "name": f"test_subset{subset_test}",
                "X": X_test_proc[sel_t],
                "y": y_test_proc[sel_t],
                "meta": {"ids": sel_t.tolist()},
            }
        )

    return X_train_proc, y_train_proc, test_sets
