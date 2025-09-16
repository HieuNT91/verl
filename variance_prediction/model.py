from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score

from .config import GPRConfig


"""Gaussian processes regression."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from numbers import Integral, Real
from operator import itemgetter

import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve_triangular

from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context, clone
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import validate_data
from sklearn.gaussian_process.kernels import RBF, Kernel

GPR_CHOLESKY_LOWER = True


class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of [RW2006]_.

    In addition to standard scikit-learn estimator API,
    :class:`GaussianProcessRegressor`:

    * allows prediction without prior fitting (based on the GP prior)
    * provides an additional method `sample_y(X)`, which evaluates samples
      drawn from the GPR (prior or posterior) at given inputs
    * exposes a method `log_marginal_likelihood(theta)`, which can be used
      externally for other ways of selecting hyperparameters, e.g., via
      Markov chain Monte Carlo.

    To learn the difference between a point-estimate approach vs. a more
    Bayesian modelling approach, refer to the example entitled
    :ref:`sphx_glr_auto_examples_gaussian_process_plot_compare_gpr_krr.py`.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel ``ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(1.0, length_scale_bounds="fixed")`` is used as default. Note that
        the kernel hyperparameters are optimized during fitting unless the
        bounds are marked as "fixed".

    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        It can also be interpreted as the variance of additional Gaussian
        measurement noise on the training observations. Note that this is
        different from using a `WhiteKernel`. If an array is passed, it must
        have the same number of entries as the data used for fitting and is
        used as datapoint-dependent noise level. Allowing to specify the
        noise level directly as a parameter is mainly for convenience and
        for consistency with :class:`~sklearn.linear_model.Ridge`.
        For an example illustrating how the alpha parameter controls
        the noise variance in Gaussian Process Regression, see
        :ref:`sphx_glr_auto_examples_gaussian_process_plot_gpr_noisy_targets.py`.

    optimizer : "fmin_l_bfgs_b", callable or None, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func': the objective function to be minimized, which
                #   takes the hyperparameters theta as a parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the L-BFGS-B algorithm from `scipy.optimize.minimize`
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are: `{'fmin_l_bfgs_b'}`.

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that `n_restarts_optimizer == 0` implies that one
        run is performed.

    normalize_y : bool, default=False
        Whether or not to normalize the target values `y` by removing the mean
        and scaling to unit-variance. This is recommended for cases where
        zero-mean, unit-variance priors are used. Note that, in this
        implementation, the normalisation is reversed before the GP predictions
        are reported.

        .. versionchanged:: 0.23

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    n_targets : int, default=None
        The number of dimensions of the target values. Used to decide the number
        of outputs when sampling from the prior distributions (i.e. calling
        :meth:`sample_y` before :meth:`fit`). This parameter is ignored once
        :meth:`fit` has been called.

        .. versionadded:: 1.3

    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        Feature vectors or other representations of training data (also
        required for prediction).

    y_train_ : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values in training data (also required for prediction).

    kernel_ : kernel instance
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters.

    L_ : array-like of shape (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``.

    alpha_ : array-like of shape (n_samples,)
        Dual coefficients of training data points in kernel space.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianProcessClassifier : Gaussian process classification (GPC)
        based on Laplace approximation.

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0, 592.1]), array([316.6, 316.6]))
    """

    _parameter_constraints: dict = {
        "kernel": [None, Kernel],
        "alpha": [Interval(Real, 0, None, closed="left"), np.ndarray],
        "optimizer": [StrOptions({"fmin_l_bfgs_b"}), callable, None],
        "n_restarts_optimizer": [Interval(Integral, 0, None, closed="left")],
        "normalize_y": ["boolean"],
        "copy_X_train": ["boolean"],
        "n_targets": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        kernel=None,
        *,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        n_targets=None,
        random_state=None,
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.n_targets = n_targets
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = validate_data(
            self,
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=ensure_2d,
            dtype=dtype,
        )

        n_targets_seen = y.shape[1] if y.ndim > 1 else 1
        if self.n_targets is not None and n_targets_seen != self.n_targets:
            raise ValueError(
                "The number of targets seen in `y` is different from the parameter "
                f"`n_targets`. Got {n_targets_seen} != {self.n_targets}."
            )

        shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
        self._y_train_mean = np.zeros(shape=shape_y_stats)
        self._y_train_std = np.ones(shape=shape_y_stats)

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel_(self.X_train_)
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        return self

    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model.

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution at query points.

        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution at query points.
            Only returned when `return_cov` is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )

        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        X = validate_data(self, X, ensure_2d=ensure_2d, dtype=dtype, reset=False)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = RBF(1.0, length_scale_bounds="fixed")
            else:
                kernel = self.kernel

            n_targets = self.n_targets if self.n_targets is not None else 1
            y_mean = np.zeros(shape=(X.shape[0], n_targets)).squeeze()

            if return_cov:
                y_cov = kernel(X)
                if n_targets > 1:
                    y_cov = np.repeat(
                        np.expand_dims(y_cov, -1), repeats=n_targets, axis=-1
                    )
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                if n_targets > 1:
                    y_var = np.repeat(
                        np.expand_dims(y_var, -1), repeats=n_targets, axis=-1
                    )
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans @ self.alpha_

            # if y_mean has shape (n_samples, 1), reshape to (n_samples,)
            if y_mean.ndim > 1 and y_mean.shape[1] == 1:
                y_mean = np.squeeze(y_mean, axis=1)

            # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
            V = solve_triangular(
                self.L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
            )

            if return_cov:
                # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
                y_cov = self.kernel_(X) - V.T @ V

                # if y_cov has shape (n_samples, n_samples, 1), reshape to
                # (n_samples, n_samples)
                if y_cov.shape[2] == 1:
                    y_cov = np.squeeze(y_cov, axis=2)

                return y_mean, y_cov
            elif return_std:
                # Compute variance of predictive distribution
                # Use einsum to avoid explicitly forming the large matrix
                # V^T @ V just to extract its diagonal afterward.
                y_var = self.kernel_.diag(X).copy()
                y_var -= np.einsum("ij,ji->i", V.T, V)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn(
                        "Predicted variances smaller than 0. "
                        "Setting those variances to 0."
                    )
                    y_var[y_var_negative] = 0.0

                # if y_var has shape (n_samples, 1), reshape to (n_samples,)
                if y_var.shape[1] == 1:
                    y_var = np.squeeze(y_var, axis=1)

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def sample_y(self, X, n_samples=1, random_state=0):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.

        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.

        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
        rng = check_random_state(random_state)

        y_mean, y_cov = self.predict(X, return_cov=True)
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = [
                rng.multivariate_normal(
                    y_mean[:, target], y_cov[..., target], n_samples
                ).T[:, np.newaxis]
                for target in range(y_mean.shape[1])
            ]
            y_samples = np.hstack(y_samples)
        return y_samples

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from scipy.linalg import solve_triangular

class ConditionedKernel(Kernel):
    """k_t(x,y) = k(x,y) - k(x,X_prev) (K_prev+σ²I)^{-1} k(X_prev,y)"""
    def __init__(self, base_kernel, X_prev, L_prev, sigma2):
        self.base_kernel = base_kernel
        self.X_prev = np.asarray(X_prev)
        self.L_prev = L_prev          # Cholesky of K_prev + sigma2 I
        self.sigma2 = sigma2

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)

        Kxy = self.base_kernel(X, Y)                   # k(X,Y)
        KxXp = self.base_kernel(X, self.X_prev)        # k(X, X_prev)
        KpY  = self.base_kernel(self.X_prev, Y)        # k(X_prev, Y)

        # Solve (K_prev+σ²I)^{-1} KpY using stored Cholesky
        # First forward: L_prev * Z = KpY
        Z = solve_triangular(self.L_prev, KpY, lower=True, check_finite=False)
        # Then backward: (L_prev^T) * W = Z  =>  W = (K_prev+σ²I)^{-1} KpY
        W = solve_triangular(self.L_prev.T, Z, lower=False, check_finite=False)

        Kcorr = KxXp @ W
        Kcond = Kxy - Kcorr

        if eval_gradient:
            # Keep base hyperparams fixed (no gradients) to preserve equivalence.
            # If you want to re-optimize, you must implement proper chain-rule grads here.
            return Kcond, np.zeros((Kcond.shape[0], Kcond.shape[1], 0))
        return Kcond

    def diag(self, X):
        # diag(k_t(X,X)) = diag(k(X,X)) - rowwise diag( KxXp @ W ) with Y=X
        X = np.atleast_2d(X)
        Kxx_diag = self.base_kernel.diag(X)
        KxXp = self.base_kernel(X, self.X_prev)        # n×n_prev
        KpX  = self.base_kernel(self.X_prev, X)        # n_prev×n

        Z = solve_triangular(self.L_prev, KpX, lower=True, check_finite=False)
        W = solve_triangular(self.L_prev.T, Z, lower=False, check_finite=False)  # n_prev×n
        # correction diagonal is sum over n_prev of KxXp * W (elementwise) per row
        corr_diag = np.einsum("ij,ji->i", KxXp, W)
        return Kxx_diag - corr_diag

    def is_stationary(self):
        # After conditioning, it's no longer stationary in general.
        return False
    
class PosteriorMean:
    def __init__(self, base_kernel, X_prev, alpha_prev):
        self.base_kernel = base_kernel
        self.X_prev = np.asarray(X_prev)
        self.alpha_prev = np.asarray(alpha_prev)  # (n_prev, ) or (n_prev, T)

    def __call__(self, X):
        K = self.base_kernel(np.atleast_2d(X), self.X_prev)  # n×n_prev
        return K @ self.alpha_prev


from sklearn.utils.validation import check_is_fitted

class MeanShiftedGPR(GaussianProcessRegressor):
    def __init__(self, mean_fn=None, **kwargs):
        super().__init__(**kwargs)
        self.mean_fn = mean_fn

    def fit(self, X, y):
        # Fit on residuals: r = y - m_t(X)
        mX = 0 if self.mean_fn is None else np.asarray(self.mean_fn(X))
        return super().fit(X, y - mX)

    def predict(self, X, return_std=False, return_cov=False):
        base = super().predict(X, return_std=return_std, return_cov=return_cov)
        mX = 0 if self.mean_fn is None else np.asarray(self.mean_fn(X))
        if return_cov:
            mu, cov = base
            return mu + mX, cov
        if return_std:
            mu, std = base
            return mu + mX, std
        return base + mX
    
# Fit once on (X_old, y_old) with base kernel k and noise σ²
# gpr_t = GaussianProcessRegressor(kernel=k_base, alpha=sigma2, normalize_y=False, optimizer=None)
# gpr_t.fit(X_old, y_old)

# # Extract posterior state
# X_prev    = gpr_t.X_train_
# L_prev    = gpr_t.L_
# alpha_prev= gpr_t.alpha_           # (K_old + σ²I)^(-1) y_old
# k_base    = gpr_t.kernel_
# sigma2    = gpr_t.alpha            # noise used at step t
# mean_t   = PosteriorMean(k_base, X_prev, alpha_prev)          # m_t(x)
# kern_t   = ConditionedKernel(k_base, X_prev, L_prev, sigma2)  # k_t(x,x')

# gpr_tp1  = MeanShiftedGPR(mean_fn=mean_t, kernel=kern_t, alpha=sigma2,
#                           normalize_y=False, optimizer=None)  # keep hyperparams fixed
# gpr_tp1.fit(X_new, y_new)   # internally fits on residuals y_new - m_t(X_new)

# # Predictions at any X_*
# mu, std = gpr_tp1.predict(X_star, return_std=True)

@dataclass
class PredictionResult:
    y_pred: np.ndarray
    y_std: Optional[np.ndarray]
    metrics: Dict[str, float]
    examples: Tuple[np.ndarray, np.ndarray, np.ndarray]


class GPRAccuracyPredictor:
    """Gaussian Process Regression-based accuracy predictor.

    Fit on (X_train, y_train) and predict variance for X_test.
    """

    def __init__(self, config: Optional[GPRConfig] = None):
        self.config = config or GPRConfig()
        # Keep a base kernel to use across steps (hyperparams fixed unless re-optimized externally)
        self._base_kernel = self.config.build_kernel()
        self.model = None  # type: ignore
        self._is_fit = False

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    prev_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    prev_model: Optional[Any] = None,
    ) -> "GPRAccuracyPredictor":
        """Fit GP.

        If prev_data=(X_prev, y_prev) is provided, use the previous step's
        posterior (under the base kernel) as the prior for this step by
        constructing a mean function and a conditioned kernel, and then fit a
        MeanShiftedGPR on residuals y - m_prev(X).

    Alternatively, if a fitted `prev_model` with attributes
    (X_train_, L_, alpha_, kernel_) is provided (e.g., a
    sklearn GaussianProcessRegressor trained on all previous data with the
    same base kernel and noise), reuse it directly to build the prior,
    avoiding a refit on prev_data.
        """
        # Optionally map targets to unbounded range via logit (remain consistent across steps)
        y_fit = self._logit(y_train, eps=self.config.logit_eps) if self.config.use_logit else y_train

        if prev_data is None and prev_model is None:
            # First step: standard GP fit with the base kernel
            self.model = MeanShiftedGPR(
                kernel=clone(self._base_kernel),
                n_restarts_optimizer=self.config.n_restarts_optimizer,
                alpha=self.config.alpha,
                normalize_y=self.config.normalize_y,
                random_state=self.config.random_state,
            )
            self.model.fit(X_train, y_fit)
        else:
            # Build prior using either provided prev_model state or by fitting on prev_data
            sigma2 = self.config.alpha
            if prev_model is not None:
                # Expect a fitted model with GP posterior state
                required_attrs = ["X_train_", "L_", "alpha_", "kernel_"]
                if not all(hasattr(prev_model, a) for a in required_attrs):
                    raise ValueError("prev_model does not expose required GP state (X_train_, L_, alpha_, kernel_)")
                Xp = prev_model.X_train_
                Lp = prev_model.L_
                alphap = prev_model.alpha_
                k_base = prev_model.kernel_
            else:
                # Build prior from previous data using the base kernel
                X_prev, y_prev = prev_data  # type: ignore
                y_prev_fit = self._logit(y_prev, eps=self.config.logit_eps) if self.config.use_logit else y_prev
                # Fit a GP on previous data with fixed hyperparams to extract posterior state
                gp_prev = GaussianProcessRegressor(
                    kernel=clone(self._base_kernel),
                    alpha=sigma2,
                    normalize_y=self.config.normalize_y,
                    optimizer=None,  # keep hyperparameters fixed here
                    n_restarts_optimizer=0,
                    random_state=self.config.random_state,
                )
                gp_prev.fit(X_prev, y_prev_fit)
                Xp = gp_prev.X_train_
                Lp = gp_prev.L_
                alphap = gp_prev.alpha_
                k_base = gp_prev.kernel_

            mean_t = PosteriorMean(k_base, Xp, alphap)
            kern_t = ConditionedKernel(k_base, Xp, Lp, sigma2)

            # Fit residuals wrt mean_t using the conditioned kernel; keep hyperparams fixed
            self.model = MeanShiftedGPR(
                mean_fn=mean_t,
                kernel=kern_t,
                alpha=sigma2,
                normalize_y=self.config.normalize_y,
                optimizer=None,
                random_state=self.config.random_state,
            )
            self.model.fit(X_train, y_fit)

        self._is_fit = True
        return self

    def predict(self, X_test: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self._is_fit:
            raise RuntimeError("Model must be fit before prediction.")
        if return_std:
            y_mean, y_std = self.model.predict(X_test, return_std=True)
            if self.config.use_logit:
                # Inverse-transform to prob space
                y_pred = self._sigmoid(y_mean)
                if self.config.transform_std_to_prob and y_std is not None:
                    # delta-method approximation for std in prob space
                    y_std = y_pred * (1.0 - y_pred) * y_std
                return y_pred, y_std
            return y_mean, y_std
        y_mean = self.model.predict(X_test)
        if self.config.use_logit:
            return self._sigmoid(y_mean), None
        return y_mean, None

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        meta_info: Optional[Dict[str, Any]] = None,
        n_examples: int = 5,
    ) -> PredictionResult:
        y_pred, y_std = self.predict(X_test, return_std=True)
        metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }

        # example selection
        idx = np.arange(len(y_test))
        if meta_info and "ids" in meta_info:
            ids = np.array(meta_info["ids"])  # optional alignment check
            if len(ids) == len(y_test):
                idx = np.arange(len(y_test))
        sel = idx[: min(n_examples, len(idx))]
        examples = (y_test[sel], y_pred[sel], None if y_std is None else y_std[sel])
        return PredictionResult(y_pred=y_pred, y_std=y_std, metrics=metrics, examples=examples)
