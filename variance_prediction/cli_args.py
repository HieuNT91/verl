from __future__ import annotations

import argparse
from typing import Optional


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Variance prediction experiments (GPR and baselines)")
    # Data
    p.add_argument(
        "--dataset",
        choices=["california", "timedata"],
        default="timedata",
        help="Dataset to use",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    p.add_argument("--subsample-train", type=int, default=756, help="Subsample size for training (california only)")
    p.add_argument("--subset-test", type=int, default=300, help="Optional test subset size (0 to disable)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--scale-features", action="store_true", help="Standardize features")

    # TimeDataSimulator specific (used only when --dataset=timedata)
    p.add_argument(
        "--embedding-path",
        type=str,
        default="/home/hieunt/verl/data/embedding_data/embeddings_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398.npy",
        help="Path to embeddings .npy (timedata)",
    )
    p.add_argument(
        "--pairwise-path",
        type=str,
        default="/home/hieunt/verl/data/embedding_data/pairwise_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398_matrix.npy",
        help="Path to pairwise matrix .npy (timedata)",
    )
    p.add_argument(
        "--indices-path",
        type=str,
        default="/home/hieunt/verl/data/embedding_data/indices_Qwen-Qwen3-Embedding-0.6B_fixprompt-dapo-math-17k_17398.json",
        help="Path to indices .json (timedata)",
    )
    p.add_argument(
        "--regression-path",
        type=str,
        default="/home/hieunt/verl/data/regression_data/allo_grpo_4e/per_question_statistics_latest.json",
        help="Path to regression JSON (timedata)",
    )
    # Step selection: either a single --step or a list via --steps
    p.add_argument("--step", type=int, default=100, help="Current step (timedata)")
    p.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of steps to run (timedata). If provided, overrides --step",
    )
    p.add_argument("--window-size", type=int, default=2, help="Window size (timedata)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size per step (timedata)")
    p.add_argument(
        "--target-key",
        type=str,
        default="mean_acc_per_epoch",
        help="Target key in regression JSON (timedata)",
    )

    # GPR config
    p.add_argument("--use-logit", action="store_true", help="Train GPR on logit(y) and invert with sigmoid")
    p.add_argument("--length-scale", type=float, default=2, help="RBF length scale")
    p.add_argument(
        "--length-scales",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of RBF length scales to try (e.g., --length-scales 0.5 1 2). If provided, overrides --length-scale",
    )
    p.add_argument("--noise-level", type=float, default=1, help="WhiteKernel noise level")
    p.add_argument("--n-restarts", type=int, default=4, help="Optimizer restarts for GPR")

    # Output control
    p.add_argument("--save-experiments", action="store_true", help="If set, save experiments to results folder")
    return p


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    return get_parser().parse_args(args=args)
