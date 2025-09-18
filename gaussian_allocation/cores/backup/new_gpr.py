from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import warnings

# Ensure project root is on sys.path so local package is importable when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from variance_prediction.backup.new_benchmark import run_new_benchmark
from variance_prediction.cli_args import parse_args
from variance_prediction.time_data_simulator import TimeDataSimulator, TimeDataSimulatorConfig


def main() -> None:
    args = parse_args()

    if args.dataset != "timedata":
        raise ValueError("new_gpr.py currently supports --dataset=timedata only")

    cfg = TimeDataSimulatorConfig(
        embedding_path=args.embedding_path,
        pairwise_path=args.pairwise_path,
        indices_path=args.indices_path,
        regression_json_path=args.regression_path,
    )
    sim = TimeDataSimulator(cfg)

    # Steps: fixed 1..50 as requested
    steps = list(range(1, 51))

    prev_model: Optional[Any] = None
    all_results: Dict[int, List[Dict[str, Any]]] = {}

    def _filter_finite(X: np.ndarray, y: np.ndarray, step_val: int):
        invalid = ~np.isfinite(y)
        n_invalid = int(invalid.sum())
        if n_invalid > 0:
            warnings.warn(
                f"Found {n_invalid} non-finite target values at step={step_val}; filtering them out.",
                RuntimeWarning,
            )
        mask = ~invalid
        return X[mask], y[mask]

    for step in steps:
        out = sim.get_train_test_features(
            step=step,
            window_size=args.window_size,
            batch_size=args.batch_size,
            target_key=args.target_key,
        )

        X_train_raw, P_train, y_train_raw = out["train"]["X"], out["train"]["P"], out["train"]["y"]
        X_test_raw, P_test, y_test_raw = out["test"]["X"], out["test"]["P"], out["test"]["y"]

        X_train, y_train = _filter_finite(X_train_raw, y_train_raw, step)
        X_test, y_test = _filter_finite(X_test_raw, y_test_raw, step)

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

        # Run one-step benchmark, reusing prev_model for prior
        results, prev_model = run_new_benchmark(
            X_train,
            y_train,
            test_sets=test_sets,
            prev_model=prev_model,
            config_overrides={
                "n_restarts_optimizer": args.n_restarts,
                "length_scale": args.length_scale,
                "noise_level": args.noise_level,
                "use_logit": True,
                "logit_eps": 1e-4,
                "transform_std_to_prob": True,
            },
        )

        all_results[step] = results

        print(f"\nStep {step} results:")
        for r in results:
            print(r["test_name"], r["metrics"])

        # optional saving per step
        if args.save_experiments:
            run_tag = f"timedata_{step}_{args.window_size}_newgp"
            run_dir = os.path.join("results", "variance_prediction", run_tag)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump({"gpr_chained": results}, f, indent=2)
            # write examples
            for res in results:
                examples = res.get("examples")
                if not examples:
                    continue
                test_name = res.get("test_name", "test")
                out_path = os.path.join(run_dir, f"gpr_chained__{test_name}.csv")
                y_true = examples["y_true"]
                y_pred = examples["y_pred"]
                y_std = examples["y_std"]
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write("y_true,y_pred,y_std\n")
                    for i in range(len(y_true)):
                        ys = "" if y_std is None else y_std[i]
                        fh.write(f"{y_true[i]},{y_pred[i]},{ys}\n")


if __name__ == "__main__":
    main()
