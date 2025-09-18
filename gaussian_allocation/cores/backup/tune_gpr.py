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
from variance_prediction.time_data_simulator import TimeDataSimulator, TimeDataSimulatorConfig
from variance_prediction.cli_args import parse_args


def main() -> None:
    # No external baselines needed for tuning

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

            # Tune length_scale for logit GPR with transformed std
            # Priority: CLI --length-scales > env LENGTH_SCALES > single --length-scale
            length_scales = list(args.length_scales)

            print(f"\nDataset: {args.dataset} | step={step} | window={args.window_size}")
            print("Tuning length_scale values:", length_scales)

            tuned_results: List[Dict[str, Any]] = []
            for ls in length_scales:
                res = run_benchmark(
                    X_train,
                    y_train,
                    test_sets=test_sets,
                    config_overrides={
                        "n_restarts_optimizer": args.n_restarts,
                        "length_scale": ls,
                        "noise_level": args.noise_level,
                        "use_logit": True,
                        "logit_eps": 1e-4,
                        "transform_std_to_prob": True,
                    },
                )
                tuned_results.append({"length_scale": ls, "results": res})
                # Print summary line per test
                for r in res:
                    print(f"ls={ls} | {r['test_name']} => {r['metrics']}")


            # Saving experiments (folder per run, files per method)
            if args.save_experiments:
                run_tag = f"length_scale_tuning_n_restart_0/timedata_{step}_{args.window_size}_tune"
                run_dir = os.path.join("results", "variance_prediction", run_tag)
                os.makedirs(run_dir, exist_ok=True)

                # Save summary metrics
                with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump({"gpr_tuned": tuned_results}, f, indent=2)

                # Save prediction tables for each tuned length_scale
                for entry in tuned_results:
                    ls = entry["length_scale"]
                    res_list = entry["results"]
                    for res in res_list:
                        examples = res.get("examples")
                        if examples is None:
                            continue
                        test_name = res.get("test_name", "test")
                        out_path = os.path.join(run_dir, f"gpr_logit_ls{ls}__{test_name}.csv")
                        y_true = examples["y_true"]
                        y_pred = examples["y_pred"]
                        y_std = examples["y_std"]
                        with open(out_path, "w", encoding="utf-8") as fh:
                            fh.write("y_true,y_pred,y_std\n")
                            for i in range(len(y_true)):
                                ys = "" if y_std is None else y_std[i]
                                fh.write(f"{y_true[i]},{y_pred[i]},{ys}\n")

                print(f"Saved experiments to {run_dir}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")



if __name__ == "__main__":
    main()
