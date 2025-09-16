from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _logit(p, eps=1e-6): 
    p = np.clip(p, eps, 1.0 - eps) 
    return np.log(p / (1.0 - p))

def make_bounded_regression(n_samples=512,
                            n_features=128,
                            rho=0.5,          # feature correlation (AR(1)-style)
                            n_informative=5,  # how many features truly matter
                            noise_std=0.5,    # latent Gaussian noise (on g)
                            beta_k=50.0,      # Beta concentration for y-noise (larger = less noise)
                            seed=42):
    """
    Returns:
      X: (n_samples, n_features)
      y: (n_samples,) in [0,1]
      info: dict with latent f, weights, and diagnostics
    """
    rng = np.random.default_rng(seed)

    # --- Correlated Gaussian features X ~ N(0, Sigma) with Toeplitz structure ---
    # Sigma_ij = rho^|i-j|
    idx = np.arange(n_features)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n_samples, n_features)) @ L.T

    # --- Sparse informative weights for linear part ---
    w = np.zeros(n_features)
    informative_idx = rng.choice(n_features, size=n_informative, replace=False)
    w[informative_idx] = rng.normal(0, 1.0, size=n_informative)

    # --- Latent signal: linear + mild nonlinear interactions ---
    f = X @ w
    if n_features >= 5:
        f += 0.5*np.sin(X[:, 0]) + 0.3*X[:, 1]*X[:, 2] + 0.5*np.tanh(X[:, 3] - 0.5*X[:, 4])

    # --- Add Gaussian noise in latent (unbounded) space ---
    g = f + rng.normal(0, noise_std, size=n_samples)

    # --- Map to [0,1] via sigmoid ---
    p = 1.0 / (1.0 + np.exp(-g))
    p = np.clip(p, 1e-6, 1 - 1e-6)

    # --- Optional: add Beta noise on probability scale for continuous y in [0,1] ---
    # Var(y | p) = p(1-p)/(beta_k+1); set beta_k large for tight noise around p
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- transform targets to logit space ---
y_train_logit = _logit(y_train)
y_test_logit  = _logit(y_test)   # for diagnostics only

# --- kernel & model ---
kernel = (
    C(1.0, (1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-6, 1e1))
)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=False,         # we already centered via logit (unbounded)
    n_restarts_optimizer=5,    # bump this if you can afford it
    alpha=1e-10,               # tiny nugget for stability
    random_state=42,
)

# --- fit on logit targets ---
gpr.fit(X_train, y_train_logit)

# --- predict in logit space, then map back with sigmoid ---
y_mean_logit, y_std_logit = gpr.predict(X_test, return_std=True)
y_pred = _sigmoid(y_mean_logit)

# --- metrics in original [0,1] space ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (prob space): {mse:.4f}")
print(f"R^2 Score (prob space):         {r2:.4f}")

# --- (optional) diagnostics in logit space ---
logit_mse = mean_squared_error(y_test_logit, y_mean_logit)
print(f"MSE (logit space):               {logit_mse:.4f}")

# --- (optional) approximate predictive std in prob space via delta method ---
# var_p ≈ (σ(m)*(1-σ(m)))^2 * var_logit  => std_p ≈ σ(m)*(1-σ(m)) * std_logit
p = y_pred
y_std_prob_approx = p * (1.0 - p) * y_std_logit