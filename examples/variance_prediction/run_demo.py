from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

# Ensure project root is on sys.path so local package is importable when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from variance_prediction.data import make_bounded_regression
from variance_prediction.model import GPRVariancePredictor
from variance_prediction.benchmark import run_benchmark
from variance_prediction.config import GPRConfig


def simple_mean_baseline(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    mean_val = float(np.mean(y_train))
    y_pred = np.full(shape=(len(X_test),), fill_value=mean_val)
    y_std = np.full(shape=(len(X_test),), fill_value=float(np.std(y_train)))
    return y_pred, y_std


def main() -> None:
    # Create synthetic training data
    X, y, info = make_bounded_regression(
        n_samples=512, n_features=128, rho=0.5, n_informative=8, noise_std=0.5, beta_k=60.0, seed=42
    )
    # Split
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    # Build multiple test sets (e.g., different seeds)
    test_sets: List[Dict[str, Any]] = [
        {"name": "val", "X": X_val, "y": y_val, "meta": {"ids": list(range(len(y_val)))}}
    ]
    for s in [7, 100, 2025]:
        Xt, yt, _ = make_bounded_regression(
            n_samples=200, n_features=128, rho=0.5, n_informative=8, noise_std=0.5, beta_k=60.0, seed=s
        )
        test_sets.append({"name": f"seed_{s}", "X": Xt, "y": yt, "meta": {"ids": list(range(len(yt)))}})

    # Variant A: raw GPR
    results_raw = run_benchmark(
        X_train,
        y_train,
        test_sets=test_sets,
        config_overrides={"n_restarts_optimizer": 3, "length_scale": 1.0, "noise_level": 0.1, "use_logit": False},
    )

    # Variant B: logit-transformed GPR
    results_logit = run_benchmark(
        X_train,
        y_train,
        test_sets=test_sets,
        config_overrides={
            "n_restarts_optimizer": 3,
            "length_scale": 1.0,
            "noise_level": 0.1,
            "use_logit": True,
            "logit_eps": 1e-6,
            "transform_std_to_prob": True,
        },
    )

    # --- Baselines ---
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    # Baseline C1: simple mean on all test sets
    baseline_mean_results: List[Dict[str, Any]] = []
    for ts in test_sets:
        y_pred_mean, y_std_mean = simple_mean_baseline(X_train, y_train, ts["X"])  # noqa: F841
        metrics = {
            "mse": float(mean_squared_error(ts["y"], y_pred_mean)),
            "r2": float(r2_score(ts["y"], y_pred_mean)),
        }
        baseline_mean_results.append({"test_name": ts["name"], "metrics": metrics})

    # Baseline C2: RandomForestRegressor (popular regression method)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    baseline_rf_results: List[Dict[str, Any]] = []
    for ts in test_sets:
        y_pred_rf = rf.predict(ts["X"])  # may exceed [0,1], clip for fair compare
        y_pred_rf = np.clip(y_pred_rf, 0.0, 1.0)
        metrics = {
            "mse": float(mean_squared_error(ts["y"], y_pred_rf)),
            "r2": float(r2_score(ts["y"], y_pred_rf)),
        }
        baseline_rf_results.append({"test_name": ts["name"], "metrics": metrics})

    # Baseline C3: LinearRegression (simple/popular baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    baseline_lr_results: List[Dict[str, Any]] = []
    for ts in test_sets:
        y_pred_lr = lr.predict(ts["X"])  # may exceed [0,1], clip for fair compare
        y_pred_lr = np.clip(y_pred_lr, 0.0, 1.0)
        metrics = {
            "mse": float(mean_squared_error(ts["y"], y_pred_lr)),
            "r2": float(r2_score(ts["y"], y_pred_lr)),
        }
        baseline_lr_results.append({"test_name": ts["name"], "metrics": metrics})

    # Print compact results
    print("Benchmark results (RAW GPR, summary):")
    for r in results_raw:
        print(r["test_name"], r["metrics"])

    print("\nBenchmark results (LOGIT GPR, summary):")
    for r in results_logit:
        print(r["test_name"], r["metrics"])

    print("\nBaseline (MEAN), summary:")
    for r in baseline_mean_results:
        print(r["test_name"], r["metrics"])

    print("\nBaseline (RandomForest), summary:")
    for r in baseline_rf_results:
        print(r["test_name"], r["metrics"])

    print("\nBaseline (LinearRegression), summary:")
    for r in baseline_lr_results:
        print(r["test_name"], r["metrics"])

    # Print a few example predictions from the first logit-GPR test set
    if results_logit:
        ex = results_logit[0]["examples"]
        print("\nExample predictions (LOGIT GPR, first test set):")
        for i, (yt, yp, ys) in enumerate(zip(ex["y_true"], ex["y_pred"], ex["y_std"])):
            print(f"[{i}] y_true={yt:.3f} | y_pred={yp:.3f} | y_std={ys:.3f}")
            if i >= 4:
                break

    # Optional: dump full JSON
    all_results = {
        "raw": results_raw,
        "logit": results_logit,
        "baseline_mean": baseline_mean_results,
        "baseline_rf": baseline_rf_results,
        "baseline_lr": baseline_lr_results,
    }
    print("\nFull JSON:")
    print(json.dumps(all_results, indent=2))

    # Save results to a folder
    from datetime import datetime
    output_dir = os.path.join(PROJECT_ROOT, "results", "variance_prediction")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"results_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
