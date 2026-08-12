"""
Build a noise-injected real-vs-generated dataset for the physical-consistency
classifier under realistic detector noise, using bilby's aLIGO PSD.

Both real (held-out) and generated glitches are:
  1. scale_rowwise()-normalized (fake only needs this here; real already is,
     from prepare_holdout_split.py).
  2. Rescaled to the SAME fixed target SNR per class -- the mean SNR values
     already published in the paper's Gravity Spy injection table
     (Table tab:injection_snr) -- using the identical whitened-frame formula
     for both domains (glitchgan.utils.whitened_snr_scaling), so SNR/amplitude
     itself carries no information about which domain a sample came from.
     (The paper's own Gravity Spy injections instead give real glitches their
     natural, individually-varying SNR and only force generated glitches to
     the fixed class mean -- deliberately NOT replicated here, since that
     asymmetry would let a classifier trivially separate domains by SNR
     variance alone, the same failure mode as the earlier row-mean/min-max
     scaling mismatch this pipeline already had to fix.)
  3. Injected into an INDEPENDENT bilby interferometer (default H1, O3 design
     PSD) whitened noise realization per sample -- not one shared segment --
     so no single noise realization's fluctuations can become a memorizable
     "watermark" shared across every example.

Usage:
    python scripts/build_injected_dataset.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \\
        --out-npz injected_data/injected_real_vs_fake.npz
"""

import argparse
import os

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from classify_real_vs_generated import generate_fake_samples_for_class, load_generator
from glitchgan.utils import whitened_snr_scaling
from prepare_holdout_split import scale_rowwise
from real_snr_distribution import load_real_snr_per_class, sample_snr

# Mean SNR per class, from Table tab:injection_snr in the paper (used there
# for the Gravity Spy synthetic-glitch validation injections).
MEAN_SNR_PER_CLASS = {
    "Blip": 29.8,
    "Fast_Scattering": 36.4,
    "Koi_Fish": 187.2,
    "Low_Frequency_Burst": 40.3,
    "Scattered_Light": 31.5,
    "Tomte": 25.4,
    "Whistle": 27.1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a noise-injected real-vs-generated dataset, both domains "
                     "scaled to the same per-class mean SNR and injected into "
                     "independent bilby aLIGO noise realizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--holdout-npz", type=str, required=True)
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--ifo", type=str, default="H1")
    parser.add_argument("--sample-rate", type=float, default=4096.0)
    parser.add_argument("--snr-mode", type=str, default="sample", choices=["fixed", "sample"],
                        help="'fixed' uses each class's single mean SNR (Table "
                             "tab:injection_snr) for every sample. 'sample' draws each "
                             "sample's own SNR independently from the empirical "
                             "distribution of real per-class SNR values (--o3a-csv/"
                             "--o3b-csv), which is heavily right-skewed -- more realistic "
                             "than a single fixed value. Applied identically to both "
                             "real and fake samples either way.")
    parser.add_argument("--o3a-csv", type=str,
                        default="/home/tom.dooney/GravitySpy_datasets/data_o3a_high_confidence.csv",
                        help="Only used when --snr-mode sample.")
    parser.add_argument("--o3b-csv", type=str,
                        default="/home/tom.dooney/GravitySpy_datasets/data_o3b_high_confidence.csv",
                        help="Only used when --snr-mode sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-npz", type=str, required=True)
    return parser.parse_args()


def inject_batch(X, ifo, sample_rate, rng):
    """Inject each row of X into an independent whitened noise realization,
    reusing a single pre-constructed Interferometer (its PSD is loaded once;
    each call to set_strain_data_from_power_spectral_density draws a fresh,
    independent Gaussian noise realization from it)."""
    import bilby

    n_samples = X.shape[-1]
    duration = n_samples / sample_rate
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(len(X)):
        seed = int(rng.integers(0, 2**31 - 1))
        bilby.core.utils.random.seed(seed)
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sample_rate, duration=duration, start_time=0.0,
        )
        noise = ifo.whitened_time_domain_strain
        out[i] = (X[i] + noise).astype(np.float32)
    return out


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Generator checkpoint: {args.generator_checkpoint}")
    print(f"Detector: {args.ifo}  Sample rate: {args.sample_rate} Hz")

    print("Loading held-out real data...")
    d = np.load(args.holdout_npz, allow_pickle=True)
    X_real = d["X"]
    y_real_idx = np.argmax(d["y"], axis=1)
    label_order = list(d["label_order"])
    for name in label_order:
        if name not in MEAN_SNR_PER_CLASS:
            raise ValueError(f"No mean SNR entry for class '{name}' -- update "
                             f"MEAN_SNR_PER_CLASS to match label_order.")

    print("Loading generator and generating matching fake samples...")
    generator = load_generator(args.generator_checkpoint)
    fake_parts, fake_class_idx_parts = [], []
    for c, name in enumerate(label_order):
        n = int((y_real_idx == c).sum())
        fake_parts.append(
            generate_fake_samples_for_class(generator, n, c, args.num_classes, args.noise_dim, rng)
        )
        fake_class_idx_parts.append(np.full(n, c))
    X_fake = np.concatenate(fake_parts)
    class_idx_fake = np.concatenate(fake_class_idx_parts)
    print("Applying scale_rowwise() to generated samples (matching real data's scaling)...")
    X_fake = scale_rowwise(X_fake)

    # whitened_snr_scaling() assumes a flat PSD of exactly 1 (SNR^2 = 4*df*sum|h(f)|^2),
    # but bilby's whitened_time_domain_strain has unit TIME-DOMAIN variance, which
    # corresponds to a flat one-sided PSD of 2/fs, not 1. Reconciling the two:
    # SNR_true = SNR_formula * sqrt(fs/2), so dividing by sqrt(fs/2) after scaling
    # corrects the mismatch (a factor of ~45x at fs=4096) that was making injected
    # glitches drown out the noise floor relative to their nominal target SNR.
    snr_unit_correction = np.sqrt(args.sample_rate / 2.0)

    real_snr_per_class = None
    if args.snr_mode == "sample":
        print(f"\nLoading empirical per-class SNR distributions from "
             f"{args.o3a_csv} / {args.o3b_csv}...")
        real_snr_per_class = load_real_snr_per_class(args.o3a_csv, args.o3b_csv, label_order)
        for name in label_order:
            snrs = real_snr_per_class[name]
            print(f"  [{name}] n_available={len(snrs)}  mean={snrs.mean():.1f}  "
                 f"median={np.median(snrs):.1f}")

    def get_snr(name, n):
        if args.snr_mode == "fixed":
            return MEAN_SNR_PER_CLASS[name]
        return sample_snr(real_snr_per_class, name, n, rng)

    print(f"\nRescaling both domains to per-class SNR (mode={args.snr_mode})...")
    X_real_scaled = np.zeros_like(X_real, dtype=np.float32)
    for c, name in enumerate(label_order):
        mask = y_real_idx == c
        snr = get_snr(name, int(mask.sum()))
        X_real_scaled[mask] = whitened_snr_scaling(
            X_real[mask], snr, srate=int(args.sample_rate)) / snr_unit_correction
        snr_desc = f"{snr}" if args.snr_mode == "fixed" else f"mean={np.mean(snr):.1f}"
        print(f"  real  [{name}]: {mask.sum()} samples -> SNR {snr_desc}")

    X_fake_scaled = np.zeros_like(X_fake, dtype=np.float32)
    for c, name in enumerate(label_order):
        mask = class_idx_fake == c
        snr = get_snr(name, int(mask.sum()))
        X_fake_scaled[mask] = whitened_snr_scaling(
            X_fake[mask], snr, srate=int(args.sample_rate)) / snr_unit_correction
        snr_desc = f"{snr}" if args.snr_mode == "fixed" else f"mean={np.mean(snr):.1f}"
        print(f"  fake  [{name}]: {mask.sum()} samples -> SNR {snr_desc}")

    print(f"\nInjecting into independent {args.ifo} bilby noise realizations...")
    import bilby
    ifo = bilby.gw.detector.get_empty_interferometer(args.ifo)  # PSD loaded once

    print(f"  real: {len(X_real_scaled)} samples")
    X_real_injected = inject_batch(X_real_scaled, ifo, args.sample_rate, rng)
    print(f"  fake: {len(X_fake_scaled)} samples")
    X_fake_injected = inject_batch(X_fake_scaled, ifo, args.sample_rate, rng)

    np.savez(
        args.out_npz,
        X_real=X_real_injected, y_real_idx=y_real_idx,
        X_fake=X_fake_injected, y_fake_idx=class_idx_fake,
        label_order=np.array(label_order),
        mean_snr_per_class=np.array([MEAN_SNR_PER_CLASS[n] for n in label_order]),
        snr_mode=args.snr_mode,
        seed=args.seed, generator_checkpoint=args.generator_checkpoint,
        ifo=args.ifo, sample_rate=args.sample_rate,
    )
    print(f"\nSaved: {args.out_npz}")
    print(f"  X_real: {X_real_injected.shape}   X_fake: {X_fake_injected.shape}")


if __name__ == "__main__":
    main()
