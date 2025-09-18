from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Any, Tuple


def collect_runs(root: str) -> List[str]:
    runs = []
    if not os.path.isdir(root):
        return runs
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if os.path.isdir(run_dir) and os.path.isfile(os.path.join(run_dir, "summary.json")):
            runs.append(run_dir)
    return runs


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_results(run_name: str, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add_rows(method_name: str, entries: List[Dict[str, Any]]):
        for e in entries or []:
            tn = e.get("test_name", "test")
            metrics = e.get("metrics", {})
            rows.append(
                {
                    "run": run_name,
                    "method": method_name,
                    "test_name": tn,
                    "mse": metrics.get("mse", None),
                    "r2": metrics.get("r2", None),
                }
            )

    # Expect keys: gpr, gpr_raw, baseline_lr (as saved by debug_gpr)
    add_rows("gpr_logit", summary.get("gpr"))
    add_rows("gpr_raw", summary.get("gpr_raw"))
    add_rows("baseline_lr", summary.get("baseline_lr"))
    add_rows("gpr_noise0", summary.get("gpr_noise0"))
    return rows


def to_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No results found."
    # Compute column widths
    headers = ["run", "method", "test_name", "mse", "r2"]
    str_rows = []
    for r in rows:
        str_rows.append(
            [
                str(r.get("run", "")),
                str(r.get("method", "")),
                str(r.get("test_name", "")),
                f"{r.get('mse', '')}",
                f"{r.get('r2', '')}",
            ]
        )
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: List[str]) -> str:
        return " | ".join(val.ljust(widths[i]) for i, val in enumerate(vals))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    for row in str_rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize variance_prediction results")
    ap.add_argument(
        "--results-root",
        type=str,
        default=os.path.join("results", "variance_prediction"),
        help="Root folder containing run subfolders",
    )
    ap.add_argument(
        "--save-csv",
        type=str,
        default="",
        help="Optional path to write the flat table as CSV",
    )
    args = ap.parse_args()

    runs = collect_runs(args.results_root)
    all_rows: List[Dict[str, Any]] = []
    for run_dir in runs:
        run_name = os.path.basename(run_dir)
        summary_path = os.path.join(run_dir, "summary.json")
        try:
            summary = load_summary(summary_path)
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {run_dir}: failed to read summary.json ({e})")
            continue
        all_rows.extend(flatten_results(run_name, summary))

    # Print table
    print(to_table(all_rows))

    # Optional CSV export
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        with open(args.save_csv, "w", encoding="utf-8") as f:
            f.write("run,method,test_name,mse,r2\n")
            for r in all_rows:
                f.write(
                    f"{r['run']},{r['method']},{r['test_name']},{r.get('mse','')},{r.get('r2','')}\n"
                )
        print(f"Saved CSV to {args.save_csv}")


if __name__ == "__main__":
    main()
