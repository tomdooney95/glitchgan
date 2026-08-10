"""
Diagnostic: check whether the real-vs-fake classifier (classify_real_vs_generated.py)
is exploiting a trivial normalization artifact rather than genuine morphological
differences.

Real held-out samples go through prepare_holdout_split.py's scale_rowwise() --
an explicit per-sample min-max scale to [-1, 1] followed by subtracting that
sample's own mean, which guarantees EXACTLY zero row-mean (and near-exact +-1
row min/max) for every real sample. Generated samples are raw generator
output with no equivalent post-processing, so their row-mean/min/max are only
whatever the generator implicitly learned to produce. If that spread is
distinguishable from real data's mathematically exact zero-mean/+-1-saturated
signature, a classifier can get near-perfect accuracy from that alone,
without ever learning anything about glitch morphology.

Usage:
    python scripts/diagnose_real_vs_fake_stats.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_400.keras
"""

import argparse
import os

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from classify_real_vs_generated import generate_fake_samples_for_class, load_generator
from prepare_holdout_split import scale_rowwise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare basic per-sample statistics between held-out real and "
                     "generated glitches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--holdout-npz", type=str, required=True)
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summarize(X, name):
    row_min = X.min(axis=1)
    row_max = X.max(axis=1)
    row_mean = X.mean(axis=1)
    row_std = X.std(axis=1)

    print(f"\n{name} (n={len(X)}):")
    print(f"  row-mean : mean={row_mean.mean(): .6f}  std={row_mean.std():.6f}  "
          f"[{row_mean.min(): .6f}, {row_mean.max(): .6f}]")
    print(f"  row-min  : mean={row_min.mean(): .4f}  std={row_min.std():.4f}  "
          f"[{row_min.min(): .4f}, {row_min.max(): .4f}]")
    print(f"  row-max  : mean={row_max.mean(): .4f}  std={row_max.std():.4f}  "
          f"[{row_max.min(): .4f}, {row_max.max(): .4f}]")
    print(f"  row-std  : mean={row_std.mean(): .4f}  std={row_std.std():.4f}")

    frac_near_pos1 = np.isclose(row_max, 1.0, atol=1e-2).mean()
    frac_near_neg1 = np.isclose(row_min, -1.0, atol=1e-2).mean()
    print(f"  frac reaching ~+1.0 (atol=1e-2): {frac_near_pos1:.3f}")
    print(f"  frac reaching ~-1.0 (atol=1e-2): {frac_near_neg1:.3f}")

    return row_mean, row_min, row_max


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading held-out real data...")
    d = np.load(args.holdout_npz, allow_pickle=True)
    X_real = d["X"]
    y_real_idx = np.argmax(d["y"], axis=1)
    label_order = list(d["label_order"])

    print("Loading generator and generating matching fake samples...")
    generator = load_generator(args.generator_checkpoint)
    fake_parts = []
    for c, name in enumerate(label_order):
        n = int((y_real_idx == c).sum())
        fake_parts.append(
            generate_fake_samples_for_class(generator, n, c, args.num_classes, args.noise_dim, rng)
        )
    X_fake = np.concatenate(fake_parts)

    X_fake_scaled = scale_rowwise(X_fake)

    mean_real, min_real, max_real = summarize(X_real, "REAL (held-out)")
    mean_fake, min_fake, max_fake = summarize(X_fake, "FAKE (raw generator output)")
    mean_fake_sc, min_fake_sc, max_fake_sc = summarize(
        X_fake_scaled, "FAKE (scale_rowwise applied, matching real)")

    def meanonly_accuracy(mean_real, mean_fake):
        thresh = np.abs(mean_fake).mean()
        pred_real = np.abs(mean_real) < thresh
        pred_fake = np.abs(mean_fake) >= thresh
        return (pred_real.sum() + pred_fake.sum()) / (len(mean_real) + len(mean_fake)), thresh

    acc_raw, thresh_raw = meanonly_accuracy(mean_real, mean_fake)
    acc_scaled, thresh_scaled = meanonly_accuracy(mean_real, mean_fake_sc)
    print(f"\nNaive row-mean-only separability (|mean| < threshold => 'real'):")
    print(f"  vs. raw fake output:    threshold={thresh_raw:.4f}  accuracy={acc_raw:.3f}")
    print(f"  vs. scale_rowwise fake: threshold={thresh_scaled:.4f}  accuracy={acc_scaled:.3f}")
    print("If the raw-fake number alone is already high but the scaled-fake number drops "
          "toward chance, row-mean/min/max saturation was doing most of the work -- i.e. "
          "the classifier wasn't learning morphology, just this normalization mismatch.")


if __name__ == "__main__":
    main()
