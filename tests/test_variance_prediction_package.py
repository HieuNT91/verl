from __future__ import annotations

import math

import numpy as np

from variance_prediction.data import make_bounded_regression
from variance_prediction.model import GPRVariancePredictor


def test_end_to_end_demo():
    X, y, _ = make_bounded_regression(n_samples=256, n_features=32, seed=123)
    n_train = int(0.7 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    model = GPRVariancePredictor().fit(X_train, y_train)
    y_pred, y_std = model.predict(X_test)

    # Shapes
    assert y_pred.shape == y_test.shape
    assert y_std is not None and y_std.shape == y_test.shape

    # Bounds
    assert np.all((y >= 0) & (y <= 1))

    # Metrics finite
    res = model.evaluate(X_test, y_test)
    assert math.isfinite(res.metrics["mse"]) and math.isfinite(res.metrics["r2"])  # noqa: SIM115

    # Example predictions contain values
    y_true_ex, y_pred_ex, y_std_ex = res.examples
    assert len(y_true_ex) > 0 and len(y_pred_ex) > 0 and len(y_std_ex) > 0
