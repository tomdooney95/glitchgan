"""
Consolidated overview across ALL multi-class experiment results found on
disk -- every train/val/test split tried (70/10/20, 40/10/50, 30/20/50,
20/20/60, 10/20/70, ...) and every balancing target (default majority-class
count, 5000/class, ...) -- so the full landscape of results can be compared
in one table instead of one classify_multiclass.py run at a time.

Automatically discovers result directories matching a glob pattern (default:
multiclass_results*), and for each one that has at least one of the four
config subdirectories (real_natural, real_balanced, real_fake_augmented,
fake_only), prints one row of overall accuracy + 95% Wilson CI per config.

Usage:
    python scripts/summarize_all_multiclass_results.py
    python scripts/summarize_all_multiclass_results.py --glob "multiclass_results_epoch500*"
    python scripts/summarize_all_multiclass_results.py --detail   # also print per-class tables
"""

import argparse
import glob
import os

import numpy as np

CONFIGS = ["real_natural", "real_balanced", "real_fake_augmented", "fake_only"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidated overview across all multi-class experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--glob", type=str, default="multiclass_results*",
                        help="Glob pattern (relative to cwd) matching result parent "
                             "directories, each expected to contain one subdirectory "
                             "per config.")
    parser.add_argument("--detail", action="store_true",
                        help="Also print the full per-class breakdown for each "
                             "results directory (verbose -- one block per experiment).")
    return parser.parse_args()


def load_dir(results_dir):
    loaded = {}
    for config in CONFIGS:
        path = os.path.join(results_dir, config, "multiclass_results.npz")
        if os.path.exists(path):
            loaded[config] = np.load(path, allow_pickle=True)
    return loaded


def fmt_cell(d, config):
    if config not in d:
        return "--"
    r = d[config]
    acc, lo, hi = float(r["accuracy"]), float(r["ci_lo"]), float(r["ci_hi"])
    return f"{acc:.3f} [{lo:.3f},{hi:.3f}]"


def main():
    args = parse_args()

    candidates = sorted(d for d in glob.glob(args.glob) if os.path.isdir(d))
    rows = []
    for d in candidates:
        loaded = load_dir(d)
        if loaded:
            rows.append((d, loaded))

    if not rows:
        print(f"No result directories matching '{args.glob}' with any config data found.")
        return

    print("=" * 130)
    print("CONSOLIDATED OVERVIEW -- OVERALL ACCURACY [95% CI] PER CONFIG")
    print("=" * 130)
    col_w = 30
    header = f"{'Results directory':<50}" + "".join(f"{c:>{col_w}}" for c in CONFIGS)
    print(header)
    print("-" * len(header))
    for d, loaded in rows:
        n_present = ", ".join(f"{c}: n={int(loaded[c]['n'])}" for c in CONFIGS if c in loaded)
        row = f"{d:<50}"
        for config in CONFIGS:
            row += f"{fmt_cell(loaded, config):>{col_w}}"
        print(row)

    if args.detail:
        print()
        print("=" * 130)
        print("PER-EXPERIMENT DETAIL (per-class accuracy)")
        print("=" * 130)
        for d, loaded in rows:
            print(f"\n--- {d} ---")
            any_cfg = next(iter(loaded.values()))
            class_names = list(any_cfg["per_class_names"])
            header2 = f"{'Class':<22}" + "".join(f"{c:>24}" for c in CONFIGS)
            print(header2)
            for i, name in enumerate(class_names):
                row2 = f"{name:<22}"
                for config in CONFIGS:
                    if config not in loaded:
                        row2 += f"{'--':>24}"
                        continue
                    r = loaded[config]
                    acc_c = float(r["per_class_acc"][i])
                    lo_c = float(r["per_class_ci_lo"][i])
                    hi_c = float(r["per_class_ci_hi"][i])
                    n_c = int(r["per_class_n"][i])
                    row2 += f"{f'{acc_c:.2f} [{lo_c:.2f},{hi_c:.2f}] n={n_c}':>24}"
                print(row2)


if __name__ == "__main__":
    main()
