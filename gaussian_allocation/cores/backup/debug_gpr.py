from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import warnings

# Ensure project root is on sys.path so local package is importable when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from variance_prediction.benchmark import run_benchmark
from variance_prediction.model import GPRVariancePredictor
from variance_prediction.data import make_bounded_regression, make_california_prob_dataset
from cores.time_data_simulator import TimeDataSimulator, TimeDataSimulatorConfig
from variance_prediction.cli_args import parse_args


def main() -> None:
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression

    args = parse_args()

    # Choose dataset
    if args.dataset == "california":
        X_train, y_train, test_sets = make_california_prob_dataset(
            test_size=args.test_size,
            subsample_train=args.subsample_train,
            random_state=args.random_state,
            subset_test=args.subset_test,
            scale_features=args.scale_features,
        )
    elif args.dataset == "timedata":
        cfg = TimeDataSimulatorConfig(
            embedding_path=args.embedding_path,
            pairwise_path=args.pairwise_path,
            indices_path=args.indices_path,
            regression_json_path=args.regression_path,
        )
        sim = TimeDataSimulator(cfg)

        steps = args.steps if args.steps is not None else [args.step]
        for step in steps:
            out = sim.get_train_test_features(
                step=step,
                window_size=args.window_size,
                batch_size=args.batch_size,
                target_key=args.target_key,
            )

            # NaN/inf guard with warning (do not fail silently)
            def _filter_finite(X: np.ndarray, y: np.ndarray, step_val: int = step):
                invalid = ~np.isfinite(y)
                n_invalid = int(invalid.sum())
                if n_invalid > 0:
                    warnings.warn(
                        f"Found {n_invalid} non-finite target values at step={step_val}; filtering them out.",
                        RuntimeWarning,
                    )
                mask = ~invalid
                return X[mask], y[mask]

            X_train_raw, P_train, y_train_raw = out["train"]["X"], out["train"]["P"], out["train"]["y"]
            X_test_raw, P_test, y_test_raw = out["test"]["X"], out["test"]["P"], out["test"]["y"]

            X_train, y_train = _filter_finite(X_train_raw, y_train_raw)
            X_test, y_test = _filter_finite(X_test_raw, y_test_raw)

            if X_train.shape[0] == 0:
                raise ValueError(f"No training data available at step={step}; please use a larger window or different step.")

            test_sets = [
                {
                    "name": f"timedata_step{step}",
                    "X": X_test,
                    "y": y_test,
                    "meta": {"qids": out["test"]["qids"]},
                }
            ]

            # Run benchmarks for this step
            results_raw = run_benchmark(
                X_train,
                y_train,
                test_sets=test_sets,
                config_overrides={
                    "n_restarts_optimizer": args.n_restarts,
                    "length_scale": args.length_scale,
                    "noise_level": args.noise_level,
                    "use_logit": False,
                    "logit_eps": 1e-4,
                    "transform_std_to_prob": False,
                },
            )

            results = run_benchmark(
                X_train,
                y_train,
                test_sets=test_sets,
                config_overrides={
                    "n_restarts_optimizer": args.n_restarts,
                    "length_scale": args.length_scale,
                    "noise_level": args.noise_level,
                    "use_logit": True,
                    "logit_eps": 1e-4,
                    "transform_std_to_prob": True,
                },
            )

            # GPR with noise_level=0 (logit)
            results_noise0 = run_benchmark(
                X_train,
                y_train,
                test_sets=test_sets,
                config_overrides={
                    "n_restarts_optimizer": args.n_restarts,
                    "length_scale": args.length_scale,
                    "noise_level": 0.0,
                    "use_logit": True,
                    "logit_eps": 1e-4,
                    "transform_std_to_prob": True,
                },
            )

            # Baseline: LinearRegression (clip to [0,1])
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            baseline_lr_results: List[Dict[str, Any]] = []
            for ts in test_sets:
                y_pred_lr = lr.predict(ts["X"])  # may exceed [0,1]
                y_pred_lr_clipped = np.clip(y_pred_lr, 0.0, 1.0)
                metrics = {
                    "mse": float(mean_squared_error(ts["y"], y_pred_lr_clipped)),
                    "r2": float(r2_score(ts["y"], y_pred_lr_clipped)),
                }
                baseline_lr_results.append({"test_name": ts["name"], "metrics": metrics})

            print(f"\nDataset: {args.dataset} | step={step} | window={args.window_size}")
            print("Benchmark results (GPR, summary):")
            for r in results:
                print(r["test_name"], r["metrics"])

            print("\n Benchmark results (GPR Raw, summary):")
            for r in results_raw:
                print(r["test_name"], r["metrics"])

            print("\nBaseline (LinearRegression), summary):")
            for r in baseline_lr_results:
                print(r["test_name"], r["metrics"])

            print("\n Benchmark results (GPR noise=0, summary):")
            for r in results_noise0:
                print(r["test_name"], r["metrics"])


            # Saving experiments (folder per run, files per method)
            if args.save_experiments:
                run_tag = f"timedata_{step}_{args.window_size}"
                run_dir = os.path.join("results", "variance_prediction", run_tag)
                os.makedirs(run_dir, exist_ok=True)

                # Save summary metrics
                with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "gpr": results,
                        "gpr_raw": results_raw,
                        "gpr_noise0": results_noise0,
                        "baseline_lr": baseline_lr_results,
                    }, f, indent=2)

                # Save prediction tables (no limit on examples)
                def _write_examples(name: str, result_list):
                    for res in result_list:
                        examples = res.get("examples")
                        test_name = res.get("test_name", "test")
                        out_path = os.path.join(run_dir, f"{name}__{test_name}.csv")
                        if examples is None:
                            continue
                        y_true = examples["y_true"]
                        y_pred = examples["y_pred"]
                        y_std = examples["y_std"]
                        with open(out_path, "w", encoding="utf-8") as fh:
                            fh.write("y_true,y_pred,y_std\n")
                            for i in range(len(y_true)):
                                ys = "" if y_std is None else y_std[i]
                                fh.write(f"{y_true[i]},{y_pred[i]},{ys}\n")

                _write_examples("gpr_logit", results)
                _write_examples("gpr_raw", results_raw)
                _write_examples("gpr_noise0", results_noise0)

                # Baseline LR full table
                for ts in test_sets:
                    y_pred_lr = lr.predict(ts["X"])  # unclipped + clipped
                    y_pred_lr_clipped = np.clip(y_pred_lr, 0.0, 1.0)
                    out_path = os.path.join(run_dir, f"baseline_lr__{ts['name']}.csv")
                    with open(out_path, "w", encoding="utf-8") as fh:
                        fh.write("y_true,y_pred,y_pred_clipped\n")
                        for yt, yp, ypc in zip(ts["y"], y_pred_lr, y_pred_lr_clipped):
                            fh.write(f"{float(yt)},{float(yp)},{float(ypc)}\n")

                print(f"Saved experiments to {run_dir}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")



if __name__ == "__main__":
    main()
