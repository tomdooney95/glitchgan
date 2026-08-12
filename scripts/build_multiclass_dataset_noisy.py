"""
Noise-injected version of build_multiclass_dataset.py: builds the same four
7-class training-set variants (real natural/imbalanced, real bootstrap-
balanced, real+fake augmented to balance, fake-only balanced) plus a shared
real val/test split, but with every sample -- real and fake alike -- scaled
to its class's mean SNR (Table tab:injection_snr in the paper) and injected
into an independent bilby aLIGO (default H1, O3 design PSD) whitened noise
realization, using the same corrected SNR scaling as build_injected_dataset.py.

Ordering matters for the bootstrap/augmentation logic: the real training
pool, validation, and test sets are each injected ONCE per sample before any
balancing happens, so a bootstrap-duplicated sample (config 2) shares both
its underlying real glitch AND its specific noise realization with the
original -- the same duplication semantics as the clean-domain version, just
now on noise-injected data. Freshly-generated fake samples (used to fill the
augmentation deficit in config 3, and for all of config 4) are each injected
into their own independent noise realization at generation time.

classify_multiclass.py works unchanged on this dataset's output -- it just
loads whichever named X_<config>/y_<config> arrays are present.

Usage:
    python scripts/build_multiclass_dataset_noisy.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \\
        --out-npz multiclass_data/multiclass_datasets_epoch210_noisy.npz
"""

import argparse
import os

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from build_injected_dataset import MEAN_SNR_PER_CLASS, inject_batch
from classify_real_vs_generated import (
    generate_fake_samples_for_class,
    load_generator,
    stratified_train_val_test_split,
)
from glitchgan.utils import whitened_snr_scaling
from prepare_holdout_split import balance, scale_rowwise
from real_snr_distribution import load_real_snr_per_class, sample_snr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build noise-injected real/balanced/augmented/fake-only training "
                     "sets for the 7-class multi-class glitch classification experiment, "
                     "all evaluated on the same fixed, noise-injected real val/test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--holdout-npz", type=str, required=True)
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--target-per-class", type=int, default=None,
                        help="Same override as build_multiclass_dataset.py -- defaults "
                             "to the majority class's count in the real training pool.")
    parser.add_argument("--ifo", type=str, default="H1")
    parser.add_argument("--sample-rate", type=float, default=4096.0)
    parser.add_argument("--snr-mode", type=str, default="sample", choices=["fixed", "sample"],
                        help="'fixed' uses each class's single mean SNR (Table "
                             "tab:injection_snr) for every sample. 'sample' draws each "
                             "sample's own SNR independently from the empirical "
                             "distribution of real per-class SNR values (--o3a-csv/"
                             "--o3b-csv). Applied identically to real and fake samples "
                             "either way.")
    parser.add_argument("--o3a-csv", type=str,
                        default="/home/tom.dooney/GravitySpy_datasets/data_o3a_high_confidence.csv",
                        help="Only used when --snr-mode sample.")
    parser.add_argument("--o3b-csv", type=str,
                        default="/home/tom.dooney/GravitySpy_datasets/data_o3b_high_confidence.csv",
                        help="Only used when --snr-mode sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-npz", type=str, required=True)
    return parser.parse_args()


def get_snr(snr_mode, real_snr_per_class, name, n, rng):
    if snr_mode == "fixed":
        return MEAN_SNR_PER_CLASS[name]
    return sample_snr(real_snr_per_class, name, n, rng)


def snr_scale_by_class(X, y_idx, label_order, sample_rate, snr_unit_correction,
                       snr_mode, real_snr_per_class, rng):
    """Rescale each row of X to its class's SNR (fixed mean or per-sample empirical
    draw, per snr_mode), using the whitened-frame formula corrected for bilby's
    unit-time-domain-variance noise convention."""
    X_scaled = np.zeros_like(X, dtype=np.float32)
    for c, name in enumerate(label_order):
        mask = y_idx == c
        if mask.sum() == 0:
            continue
        snr = get_snr(snr_mode, real_snr_per_class, name, int(mask.sum()), rng)
        X_scaled[mask] = whitened_snr_scaling(
            X[mask], snr, srate=int(sample_rate)) / snr_unit_correction
    return X_scaled


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Generator checkpoint: {args.generator_checkpoint}")
    print(f"Detector: {args.ifo}  Sample rate: {args.sample_rate} Hz")

    print("Loading real holdout data...")
    d = np.load(args.holdout_npz, allow_pickle=True)
    X_real = d["X"]
    y_real_idx = np.argmax(d["y"], axis=1)
    label_order = list(d["label_order"])
    for name in label_order:
        if name not in MEAN_SNR_PER_CLASS:
            raise ValueError(f"No mean SNR entry for class '{name}'.")

    print(f"\nSplitting {args.train_frac:.0%}/{args.val_frac:.0%}/"
         f"{1 - args.train_frac - args.val_frac:.0%} stratified by class (seed={args.seed})...")
    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        y_real_idx, args.train_frac, args.val_frac, rng)

    X_train_pool, y_train_pool = X_real[train_idx], y_real_idx[train_idx]
    X_val, y_val = X_real[val_idx], y_real_idx[val_idx]
    X_test, y_test = X_real[test_idx], y_real_idx[test_idx]
    for c, name in enumerate(label_order):
        print(f"  [{name}] train: {(y_train_pool == c).sum()}  "
             f"val: {(y_val == c).sum()}  test: {(y_test == c).sum()}")
    print(f"  TOTAL  train: {len(X_train_pool)}  val: {len(X_val)}  test: {len(X_test)}")

    if args.target_per_class is not None:
        target = args.target_per_class
        print(f"\nBalancing target (explicit --target-per-class): {target}/class "
             f"({target * len(label_order)} total)")
    else:
        target = int(max((y_train_pool == c).sum() for c in range(len(label_order))))
        print(f"\nBalancing target (majority class count in train pool): {target}/class "
             f"({target * len(label_order)} total)")

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

    print(f"\nSNR-scaling (mode={args.snr_mode}) and injecting the real train/val/test "
         f"pools (one noise realization per sample, before any balancing)...")
    import bilby
    ifo = bilby.gw.detector.get_empty_interferometer(args.ifo)  # PSD loaded once

    X_train_pool_scaled = snr_scale_by_class(
        X_train_pool, y_train_pool, label_order, args.sample_rate, snr_unit_correction,
        args.snr_mode, real_snr_per_class, rng)
    X_train_pool_injected = inject_batch(X_train_pool_scaled, ifo, args.sample_rate, rng)

    X_val_scaled = snr_scale_by_class(
        X_val, y_val, label_order, args.sample_rate, snr_unit_correction,
        args.snr_mode, real_snr_per_class, rng)
    X_val_injected = inject_batch(X_val_scaled, ifo, args.sample_rate, rng)

    X_test_scaled = snr_scale_by_class(
        X_test, y_test, label_order, args.sample_rate, snr_unit_correction,
        args.snr_mode, real_snr_per_class, rng)
    X_test_injected = inject_batch(X_test_scaled, ifo, args.sample_rate, rng)

    # --- Config 1: real, natural (imbalanced), noise-injected ---------------
    X_real_natural = X_train_pool_injected.copy()
    y_real_natural = y_train_pool.copy()

    # --- Config 2: real, bootstrap-balanced (duplicates share their noise
    #     realization with the original, same semantics as the clean version)
    balanced_idx = balance(y_train_pool, target, rng)
    X_real_balanced = X_train_pool_injected[balanced_idx]
    y_real_balanced = y_train_pool[balanced_idx]

    # --- Config 3: real (natural, injected) + fresh fake (scaled + injected)
    #     to fill the per-class deficit -----------------------------------
    print("\nLoading generator for fake-augmentation / fake-only sets...")
    generator = load_generator(args.generator_checkpoint)

    aug_parts_X = [X_train_pool_injected.copy()]
    aug_parts_y = [y_train_pool.copy()]
    for c, name in enumerate(label_order):
        n_real_c = int((y_train_pool == c).sum())
        deficit = target - n_real_c
        if deficit > 0:
            fake_c = generate_fake_samples_for_class(
                generator, deficit, c, args.num_classes, args.noise_dim, rng)
            fake_c = scale_rowwise(fake_c)
            snr = get_snr(args.snr_mode, real_snr_per_class, name, deficit, rng)
            fake_c = whitened_snr_scaling(
                fake_c, snr, srate=int(args.sample_rate)
            ) / snr_unit_correction
            fake_c = inject_batch(fake_c, ifo, args.sample_rate, rng)
            aug_parts_X.append(fake_c)
            aug_parts_y.append(np.full(deficit, c))
        print(f"  [{name}] augmented with {max(deficit, 0)} fake samples "
             f"(real={n_real_c}, target={target})")
    X_real_fake_augmented = np.concatenate(aug_parts_X)
    y_real_fake_augmented = np.concatenate(aug_parts_y)
    perm = rng.permutation(len(X_real_fake_augmented))
    X_real_fake_augmented = X_real_fake_augmented[perm]
    y_real_fake_augmented = y_real_fake_augmented[perm]

    # --- Config 4: fake-only, balanced, scaled + injected -------------------
    fake_only_parts_X, fake_only_parts_y = [], []
    for c, name in enumerate(label_order):
        fake_c = generate_fake_samples_for_class(
            generator, target, c, args.num_classes, args.noise_dim, rng)
        fake_c = scale_rowwise(fake_c)
        snr = get_snr(args.snr_mode, real_snr_per_class, name, target, rng)
        fake_c = whitened_snr_scaling(
            fake_c, snr, srate=int(args.sample_rate)
        ) / snr_unit_correction
        fake_c = inject_batch(fake_c, ifo, args.sample_rate, rng)
        fake_only_parts_X.append(fake_c)
        fake_only_parts_y.append(np.full(target, c))
    X_fake_only = np.concatenate(fake_only_parts_X)
    y_fake_only = np.concatenate(fake_only_parts_y)
    perm = rng.permutation(len(X_fake_only))
    X_fake_only = X_fake_only[perm]
    y_fake_only = y_fake_only[perm]

    print(f"\nFinal training set sizes:")
    print(f"  real_natural:        {len(X_real_natural)}")
    print(f"  real_balanced:       {len(X_real_balanced)}")
    print(f"  real_fake_augmented: {len(X_real_fake_augmented)}")
    print(f"  fake_only:           {len(X_fake_only)}")
    print(f"  (shared) val:        {len(X_val_injected)}")
    print(f"  (shared) test:       {len(X_test_injected)}")

    np.savez(
        args.out_npz,
        X_real_natural=X_real_natural, y_real_natural=y_real_natural,
        X_real_balanced=X_real_balanced, y_real_balanced=y_real_balanced,
        X_real_fake_augmented=X_real_fake_augmented, y_real_fake_augmented=y_real_fake_augmented,
        X_fake_only=X_fake_only, y_fake_only=y_fake_only,
        X_val=X_val_injected, y_val=y_val,
        X_test=X_test_injected, y_test=y_test,
        label_order=np.array(label_order), target_per_class=target,
        mean_snr_per_class=np.array([MEAN_SNR_PER_CLASS[n] for n in label_order]),
        snr_mode=args.snr_mode,
        seed=args.seed, generator_checkpoint=args.generator_checkpoint,
        ifo=args.ifo, sample_rate=args.sample_rate,
    )
    print(f"\nSaved: {args.out_npz}")


if __name__ == "__main__":
    main()
