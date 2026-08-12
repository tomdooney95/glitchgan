"""
Quick visual check of the noise-injected real-vs-generated dataset (built by
build_injected_dataset.py): plots one real and one fake injected example per
class (or overall, if --n-classes 0) so glitch-vs-noise-floor amplitude can
be sanity-checked by eye.

Usage:
    python scripts/plot_injected_examples.py \\
        --injected-npz injected_data/injected_real_vs_fake_epoch210.npz \\
        --out-dir injected_plots
"""

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot one real and one fake injected example per class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--injected-npz", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="injected_plots")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = np.load(args.injected_npz, allow_pickle=True)
    X_real, y_real_idx = d["X_real"], d["y_real_idx"]
    X_fake, y_fake_idx = d["X_fake"], d["y_fake_idx"]
    label_order = list(d["label_order"])
    mean_snr = d["mean_snr_per_class"]

    n_classes = len(label_order)
    fig, axes = plt.subplots(n_classes, 2, figsize=(14, 2.2 * n_classes), sharex=True)

    t = np.arange(X_real.shape[-1])
    for c, name in enumerate(label_order):
        real_idx = rng.choice(np.where(y_real_idx == c)[0])
        fake_idx = rng.choice(np.where(y_fake_idx == c)[0])
        x_real, x_fake = X_real[real_idx], X_fake[fake_idx]

        ax_r, ax_f = axes[c, 0], axes[c, 1]
        ax_r.plot(t, x_real, color="black", lw=0.6)
        ax_f.plot(t, x_fake, color="black", lw=0.6)
        ax_r.set_ylabel(f"{name}\n(SNR {mean_snr[c]:.1f})", fontsize=9)
        ax_r.set_title(f"REAL: std={x_real.std():.2f} max|x|={np.abs(x_real).max():.2f}", fontsize=9)
        ax_f.set_title(f"FAKE: std={x_fake.std():.2f} max|x|={np.abs(x_fake).max():.2f}", fontsize=9)
        ax_r.axhline(0, color="grey", lw=0.3)
        ax_f.axhline(0, color="grey", lw=0.3)

    axes[-1, 0].set_xlabel("time sample")
    axes[-1, 1].set_xlabel("time sample")
    fig.suptitle("One injected example per class: real (left) vs. fake (right)", fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "injected_examples.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    print("\nPer-domain amplitude summary (std/max|x| should mostly reflect the "
         "target SNR and roughly match between real/fake within a class if the "
         "injection is well-calibrated; unit-variance noise alone gives std~1):")
    for c, name in enumerate(label_order):
        real_stds = X_real[y_real_idx == c].std(axis=1)
        fake_stds = X_fake[y_fake_idx == c].std(axis=1)
        print(f"  [{name}] (target SNR {mean_snr[c]:.1f}): "
             f"real std mean={real_stds.mean():.3f}  fake std mean={fake_stds.mean():.3f}")


if __name__ == "__main__":
    main()
