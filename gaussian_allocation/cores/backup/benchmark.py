from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .config import GPRConfig
from .model import GPRVariancePredictor


def run_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_sets: List[Dict[str, Any]],
    config_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run evaluations across multiple test sets with configurable hyperparams.

    test_sets: list of dicts, each with {"name": str, "X": np.ndarray, "y": np.ndarray, "meta": Optional[dict]}
    config_overrides: dict of fields to override in GPRConfig
    Returns a list of result dicts with metrics and example predictions per test set.
    """
    cfg = GPRConfig()
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise ValueError(f"Unknown config field: {k}, existing fields: {list(asdict(cfg).keys())}")

    predictor = GPRVariancePredictor(cfg).fit(X_train, y_train)

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
    return results
