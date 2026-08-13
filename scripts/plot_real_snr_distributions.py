"""
Plot the empirical per-class SNR distributions used by --snr-mode sample in
build_injected_dataset.py / build_multiclass_dataset_noisy.py (GravitySpy
O3a+O3b high-confidence catalogs, snr >= 15, ifo != 'V1', deduplicated by
GPStime). These are heavily right-skewed with extreme tails (e.g. Koi_Fish:
median ~123, max ~11,000), so histograms are plotted on a log-SNR axis for
readability, with mean/median marked.

Usage:
    python scripts/plot_real_snr_distributions.py \\
        --o3a-csv /home/tom.dooney/GravitySpy_datasets/data_o3a_high_confidence.csv \\
        --o3b-csv /home/tom.dooney/GravitySpy_datasets/data_o3b_high_confidence.csv \\
        --out-dir snr_distribution_plots
"""

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")
if "/Library/TeX/texbin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from real_snr_distribution import load_real_snr_per_class

LABEL_ORDER = ["Blip", "Fast_Scattering", "Koi_Fish", "Low_Frequency_Burst",
              "Scattered_Light", "Tomte", "Whistle"]

# Mean SNR per class, from Table tab:injection_snr in the paper, for reference.
MEAN_SNR_PER_CLASS = {
    "Blip": 29.8, "Fast_Scattering": 36.4, "Koi_Fish": 187.2,
    "Low_Frequency_Burst": 40.3, "Scattered_Light": 31.5, "Tomte": 25.4, "Whistle": 27.1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot empirical per-class SNR distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--o3a-csv", type=str,
                        default="/home/tom.dooney/GravitySpy_datasets/data_o3a_high_confidence.csv")
    parser.add_argument("--o3b-csv", type=str,
                        default="/home/tom.dooney/GravitySpy_datasets/data_o3b_high_confidence.csv")
    parser.add_argument("--out-dir", type=str, default="snr_distribution_plots")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading SNR distributions from {args.o3a_csv} / {args.o3b_csv}...")
    snr_per_class = load_real_snr_per_class(args.o3a_csv, args.o3b_csv, LABEL_ORDER)

    print("\nPer-class SNR summary:")
    for name in LABEL_ORDER:
        s = snr_per_class[name]
        print(f"  [{name}] n={len(s)}  mean={s.mean():.1f}  median={np.median(s):.1f}  "
             f"min={s.min():.1f}  max={s.max():.1f}")

    # --- Grid of per-class log-SNR histograms -------------------------------
    fig, axes = plt.subplots(len(LABEL_ORDER), 1, figsize=(8, 2.2 * len(LABEL_ORDER)))
    for i, name in enumerate(LABEL_ORDER):
        ax = axes[i]
        s = snr_per_class[name]
        log_s = np.log10(s)
        ax.hist(log_s, bins=60, color="#4C72B0", alpha=0.85)
        mean_log = np.log10(s.mean())
        median_log = np.log10(np.median(s))
        table_mean_log = np.log10(MEAN_SNR_PER_CLASS[name])
        ax.axvline(mean_log, color="crimson", ls="-", lw=1.3, label=f"mean={s.mean():.1f}")
        ax.axvline(median_log, color="black", ls="--", lw=1.3, label=f"median={np.median(s):.1f}")
        ax.axvline(table_mean_log, color="darkgreen", ls=":", lw=1.3,
                  label=f"paper mean={MEAN_SNR_PER_CLASS[name]:.1f}")
        ax.set_title(f"{name}  (n={len(s)})", fontsize=10)
        ax.set_xlabel("log10(SNR)", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "snr_distributions_per_class.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_path}")

    # --- Overlaid comparison, all classes on one axis (log SNR, paper style) -
    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(LABEL_ORDER)))
        for name, color in zip(LABEL_ORDER, colors):
            s = snr_per_class[name]
            log_s = np.log10(s)
            ax.hist(log_s, bins=60, histtype="step", color=color, lw=2.0,
                    label=name.replace("_", " "), density=True)
        ax.set_xlabel(r"$\log_{10}(\mathrm{SNR})$", fontsize=22)
        ax.set_ylabel("Density", fontsize=22)
        ax.tick_params(axis="both", labelsize=18)
        ax.legend(fontsize=16)
        plt.tight_layout()
        fig_path2 = os.path.join(args.out_dir, "snr_distributions_overlaid.pdf")
        fig.savefig(fig_path2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path2}")

    # --- Same comparison, linear SNR axis (reference only -- the heavy tails
    #     make this much less readable than the log version above) ----------
    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(LABEL_ORDER)))
        for name, color in zip(LABEL_ORDER, colors):
            s = snr_per_class[name]
            ax.hist(s, bins=60, histtype="step", color=color, lw=2.0,
                    label=name.replace("_", " "), density=True)
        ax.set_xlabel("SNR", fontsize=22)
        ax.set_ylabel("Density", fontsize=22)
        ax.tick_params(axis="both", labelsize=18)
        ax.legend(fontsize=16)
        plt.tight_layout()
        fig_path3 = os.path.join(args.out_dir, "snr_distributions_overlaid_linear.pdf")
        fig.savefig(fig_path3, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path3}")


if __name__ == "__main__":
    main()
