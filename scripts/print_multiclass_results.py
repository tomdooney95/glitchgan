"""
Print a consolidated summary of all four multi-class classification results
(real_natural, real_balanced, real_fake_augmented, fake_only) saved by
classify_multiclass.py, given the parent directory containing one
subdirectory per config -- including automatic significance flagging
(non-overlapping 95% Wilson CIs) against a reference config, both overall
and per class.

Usage:
    python scripts/print_multiclass_results.py --results-dir multiclass_results_epoch500
    python scripts/print_multiclass_results.py --results-dir multiclass_results_epoch500 \\
        --reference real_balanced
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
    parser.add_argument("--reference", type=str, default="real_natural", choices=CONFIGS,
                        help="Config to compare all others against for significance "
                             "flagging (non-overlapping 95%% Wilson CIs).")
    return parser.parse_args()


def cis_overlap(lo1, hi1, lo2, hi2):
    return not (hi1 < lo2 or hi2 < lo1)


def sig_marker(lo1, hi1, lo2, hi2, acc1, acc2):
    if cis_overlap(lo1, hi1, lo2, hi2):
        return ""
    return " (lower)" if acc2 < acc1 else " (higher)"


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

    ref = args.reference
    have_ref = ref in loaded
    if not have_ref:
        print(f"WARNING: reference config '{ref}' not found among loaded results -- "
             f"significance columns will be skipped.")

    print("=" * 90)
    print(f"OVERALL ACCURACY (significance vs. reference = '{ref}', 95% Wilson CI, "
         f"non-overlapping = significant)")
    print("=" * 90)
    print(f"{'Config':<22}{'Accuracy':>10}{'95% CI':>20}{'n':>8}{'vs ' + ref:>20}")
    for config in CONFIGS:
        if config not in loaded:
            continue
        d = loaded[config]
        acc, lo, hi, n = float(d["accuracy"]), float(d["ci_lo"]), float(d["ci_hi"]), int(d["n"])
        sig = ""
        if have_ref and config != ref:
            dr = loaded[ref]
            acc_r, lo_r, hi_r = float(dr["accuracy"]), float(dr["ci_lo"]), float(dr["ci_hi"])
            sig = "significant" + sig_marker(lo_r, hi_r, lo, hi, acc_r, acc) \
                if not cis_overlap(lo, hi, lo_r, hi_r) else "n.s."
        print(f"{config:<22}{acc:>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}{n:>8}{sig:>20}")

    print()
    print("=" * 90)
    print(f"PER-CLASS ACCURACY (a '*' flags a 95% CI that does not overlap "
         f"'{ref}''s CI for that same class)")
    print("=" * 90)
    class_names = list(loaded[list(loaded.keys())[0]]["per_class_names"])
    header = f"{'Class':<22}" + "".join(f"{c:>24}" for c in CONFIGS)
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name:<22}"
        ref_lo_c = ref_hi_c = None
        if have_ref:
            dr = loaded[ref]
            ref_lo_c, ref_hi_c = float(dr["per_class_ci_lo"][i]), float(dr["per_class_ci_hi"][i])
        for config in CONFIGS:
            if config not in loaded:
                row += f"{'--':>24}"
                continue
            d = loaded[config]
            acc_c = float(d["per_class_acc"][i])
            lo_c = float(d["per_class_ci_lo"][i])
            hi_c = float(d["per_class_ci_hi"][i])
            n_c = int(d["per_class_n"][i])
            flag = ""
            if have_ref and config != ref and ref_lo_c is not None:
                if not cis_overlap(lo_c, hi_c, ref_lo_c, ref_hi_c):
                    flag = "*"
            cell = f"{acc_c:.2f} [{lo_c:.2f},{hi_c:.2f}] n={n_c}{flag}"
            row += f"{cell:>24}"
        print(row)

    print()
    print("=" * 90)
    print("PROVENANCE")
    print("=" * 90)
    for config in CONFIGS:
        if config not in loaded:
            continue
        d = loaded[config]
        print(f"{config}: seed={d['seed']}  dataset={d['dataset_npz']}")


if __name__ == "__main__":
    main()
