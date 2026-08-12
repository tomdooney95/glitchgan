"""
Print a consolidated summary of all four multi-class classification results
(real_natural, real_balanced, real_fake_augmented, fake_only) saved by
classify_multiclass.py, given the parent directory containing one
subdirectory per config.

Usage:
    python scripts/print_multiclass_results.py --results-dir multiclass_results_epoch500
"""

import argparse
import os

import numpy as np

CONFIGS = ["real_natural", "real_balanced", "real_fake_augmented", "fake_only"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print a consolidated summary of multiclass classification results.",
    )
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Parent directory containing one subdirectory per config "
                             "(e.g. multiclass_results_epoch500/real_natural/...).")
    return parser.parse_args()


def main():
    args = parse_args()

    loaded = {}
    for config in CONFIGS:
        path = os.path.join(args.results_dir, config, "multiclass_results.npz")
        if not os.path.exists(path):
            print(f"[missing] {path}")
            continue
        loaded[config] = np.load(path, allow_pickle=True)

    if not loaded:
        print("No results found.")
        return

    print("=" * 70)
    print("OVERALL ACCURACY")
    print("=" * 70)
    print(f"{'Config':<22}{'Accuracy':>10}{'95% CI':>20}{'n':>8}")
    for config in CONFIGS:
        if config not in loaded:
            continue
        d = loaded[config]
        acc, lo, hi, n = float(d["accuracy"]), float(d["ci_lo"]), float(d["ci_hi"]), int(d["n"])
        print(f"{config:<22}{acc:>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}{n:>8}")

    print()
    print("=" * 70)
    print("PER-CLASS ACCURACY")
    print("=" * 70)
    class_names = list(loaded[list(loaded.keys())[0]]["per_class_names"])
    header = f"{'Class':<22}" + "".join(f"{c:>22}" for c in CONFIGS)
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name:<22}"
        for config in CONFIGS:
            if config not in loaded:
                row += f"{'--':>22}"
                continue
            d = loaded[config]
            acc_c = float(d["per_class_acc"][i])
            lo_c = float(d["per_class_ci_lo"][i])
            hi_c = float(d["per_class_ci_hi"][i])
            n_c = int(d["per_class_n"][i])
            row += f"{f'{acc_c:.2f} [{lo_c:.2f},{hi_c:.2f}] n={n_c}':>22}"
        print(row)

    print()
    print("=" * 70)
    print("PROVENANCE")
    print("=" * 70)
    for config in CONFIGS:
        if config not in loaded:
            continue
        d = loaded[config]
        print(f"{config}: seed={d['seed']}  dataset={d['dataset_npz']}")


if __name__ == "__main__":
    main()
