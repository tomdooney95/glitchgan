"""
Build the datasets for the 7-class multi-class glitch classification
experiment: real (natural/imbalanced), real (bootstrap-balanced), real+fake
(augmented to balance), and fake-only (balanced) training sets, all
evaluated on the SAME fixed, leak-proof real validation/test split.

Reuses the same real holdout set (data_holdout90/holdout_real.npz) used
throughout the real-vs-fake experiments -- these 3,151 samples were never
seen during training of the holdout-trained GlitchGAN, so it's safe to use
here too without leakage.

Split 70/10/20, stratified by class, on this holdout set:
  - 70% (~2,206 samples, class-imbalanced) is the real training pool.
  - 10% (~316 samples, fixed) is the shared validation set, used identically
    across all four training configurations for early stopping/model
    selection, so results stay directly comparable across configs.
  - 20% (~629 samples, fixed) is the shared test set, used identically
    across all four configurations for final evaluation.

Four training-set variants are built from the 70% real training pool, all
targeting the same per-class count (the majority class's count in the train
pool), so configs 2-4 are the same total size, differing only in composition:
  1. real_natural        -- real training pool as-is (class-imbalanced).
  2. real_balanced        -- real training pool, bootstrapped (oversampled
     with replacement) per class up to the target.
  3. real_fake_augmented -- real training pool, topped up with FRESH fake
     samples (not real duplicates) per class up to the target.
  4. fake_only            -- entirely synthetic, balanced to the target.

Fake samples (both augmentation and fake-only) are scale_rowwise()-
normalized, matching real data's own normalization, so no domain can be
trivially separated by a normalization mismatch.

Usage:
    python scripts/build_multiclass_dataset.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \\
        --out-npz multiclass_data/multiclass_datasets_epoch210.npz
"""

import argparse
import os

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from classify_real_vs_generated import (
    generate_fake_samples_for_class,
    load_generator,
    stratified_train_val_test_split,
)
from prepare_holdout_split import balance, scale_rowwise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build real/balanced/augmented/fake-only training sets for the "
                     "7-class multi-class glitch classification experiment, all "
                     "evaluated on the same fixed real val/test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--holdout-npz", type=str, required=True)
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-npz", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Generator checkpoint: {args.generator_checkpoint}")

    print("Loading real holdout data...")
    d = np.load(args.holdout_npz, allow_pickle=True)
    X_real = d["X"]
    y_real_idx = np.argmax(d["y"], axis=1)
    label_order = list(d["label_order"])
    print(f"  Held-out real: {X_real.shape}")

    print(f"\nSplitting 70/10/20 stratified by class (seed={args.seed})...")
    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        y_real_idx, args.train_frac, args.val_frac, rng)

    X_train_pool, y_train_pool = X_real[train_idx], y_real_idx[train_idx]
    X_val, y_val = X_real[val_idx], y_real_idx[val_idx]
    X_test, y_test = X_real[test_idx], y_real_idx[test_idx]

    for c, name in enumerate(label_order):
        print(f"  [{name}] train: {(y_train_pool == c).sum()}  "
             f"val: {(y_val == c).sum()}  test: {(y_test == c).sum()}")
    print(f"  TOTAL  train: {len(X_train_pool)}  val: {len(X_val)}  test: {len(X_test)}")

    target = int(max((y_train_pool == c).sum() for c in range(len(label_order))))
    print(f"\nBalancing target (majority class count in train pool): {target}/class "
         f"({target * len(label_order)} total)")

    # --- Config 1: real, natural (imbalanced) -------------------------------
    X_real_natural, y_real_natural = X_train_pool.copy(), y_train_pool.copy()

    # --- Config 2: real, bootstrap-balanced ---------------------------------
    balanced_idx = balance(y_train_pool, target, rng)
    X_real_balanced = X_train_pool[balanced_idx]
    y_real_balanced = y_train_pool[balanced_idx]

    # --- Config 3: real (natural) + fresh fake to fill the per-class deficit
    print("\nLoading generator for fake-augmentation / fake-only sets...")
    generator = load_generator(args.generator_checkpoint)

    aug_parts_X, aug_parts_y = [X_train_pool.copy()], [y_train_pool.copy()]
    for c, name in enumerate(label_order):
        n_real_c = int((y_train_pool == c).sum())
        deficit = target - n_real_c
        if deficit > 0:
            fake_c = generate_fake_samples_for_class(
                generator, deficit, c, args.num_classes, args.noise_dim, rng)
            fake_c = scale_rowwise(fake_c)
            aug_parts_X.append(fake_c)
            aug_parts_y.append(np.full(deficit, c))
        print(f"  [{name}] augmented with {max(deficit, 0)} fake samples "
             f"(real={n_real_c}, target={target})")
    X_real_fake_augmented = np.concatenate(aug_parts_X)
    y_real_fake_augmented = np.concatenate(aug_parts_y)
    perm = rng.permutation(len(X_real_fake_augmented))
    X_real_fake_augmented = X_real_fake_augmented[perm]
    y_real_fake_augmented = y_real_fake_augmented[perm]

    # --- Config 4: fake-only, balanced --------------------------------------
    fake_only_parts_X, fake_only_parts_y = [], []
    for c, name in enumerate(label_order):
        fake_c = generate_fake_samples_for_class(
            generator, target, c, args.num_classes, args.noise_dim, rng)
        fake_c = scale_rowwise(fake_c)
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
    print(f"  (shared) val:        {len(X_val)}")
    print(f"  (shared) test:       {len(X_test)}")

    np.savez(
        args.out_npz,
        X_real_natural=X_real_natural, y_real_natural=y_real_natural,
        X_real_balanced=X_real_balanced, y_real_balanced=y_real_balanced,
        X_real_fake_augmented=X_real_fake_augmented, y_real_fake_augmented=y_real_fake_augmented,
        X_fake_only=X_fake_only, y_fake_only=y_fake_only,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        label_order=np.array(label_order), target_per_class=target,
        seed=args.seed, generator_checkpoint=args.generator_checkpoint,
    )
    print(f"\nSaved: {args.out_npz}")


if __name__ == "__main__":
    main()
