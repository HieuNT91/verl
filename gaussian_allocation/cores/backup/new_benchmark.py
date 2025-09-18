from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import GPRConfig
from .model import GPRAccuracyPredictor


def run_new_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_sets: List[Dict[str, Any]],
    prev_model: Optional[Any] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Any]:
    """Run evaluation for a single step using GPRAccuracyPredictor.

    Reuses the previous step's posterior state via prev_model when provided.

    Returns (results, fitted_model) where fitted_model can be reused as prev_model
    for the next step.
    """
    cfg = GPRConfig()
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise ValueError(f"Unknown config field: {k}, existing fields: {list(asdict(cfg).keys())}")

    predictor = GPRAccuracyPredictor(cfg).fit(X_train, y_train, prev_model=prev_model)

    results: List[Dict[str, Any]] = []
    for ts in test_sets:
        name = ts.get("name", "test")
        X = ts["X"]
        y = ts["y"]
        meta = ts.get("meta", None)
        res = predictor.evaluate(X, y, meta_info=meta, n_examples=len(y))
        results.append(
            {
                "test_name": name,
                "config": asdict(cfg),
                "metrics": res.metrics,
                "examples": {
                    "y_true": res.examples[0].tolist(),
                    "y_pred": res.examples[1].tolist(),
                    "y_std": None if res.examples[2] is None else res.examples[2].tolist(),
                },
            }
        )

    return results, predictor.model
