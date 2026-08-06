"""
Split the raw (unbalanced) real glitch dataset into a 90% training pool and a
10% held-out test set, stratified per class, done BEFORE any oversampling.

This ordering is what makes the split leak-proof: the standard
*_balanced.npy files GlitchGAN normally trains on already contain
duplicated (oversampled) copies of underrepresented classes. Splitting
those 90/10 risks the same physical sample landing in both the training
pool and the held-out set. Splitting the raw data first, then rebalancing
only the 90% pool, guarantees every held-out sample is untouched by
training.

Built for the real-vs-generated classifier experiment requested during peer
review (physical consistency check): a separate GlitchGAN variant is
trained on the resulting 90% pool, so the 10% held-out real samples were
never seen during that model's training. This is intentionally a SEPARATE
model/dataset from the main reported GlitchGAN (trained on 100% of
available data) -- nothing about the main results changes.

Usage (run wherever has access to the raw CIT data and the glitchgan
training environment):

    python scripts/prepare_holdout_split.py \\
        --data-dir /path/to/cDVGAN_for_DeepExtractor/data \\
        --out-dir data_holdout90/ \\
        --holdout-out data_holdout90/holdout_real.npz
"""

import argparse
import os

import numpy as np

TARGET_PER_CLASS = 5000  # matches the original *_balanced.npy convention (35000 / 7 classes)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stratified 90/10 real-data split + rebalancing of the 90% training pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing the RAW (unbalanced) "
                             "glitch_GAN_samples_scaled.npy / glitch_GAN_labels.npy / "
                             "glitch_GAN_label_order.npy files.")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Where to write the rebalanced 90%% training pool, using the "
                             "same filenames glitchgan.tf.train expects "
                             "(glitch_GAN_samples_scaled_balanced.npy, etc.) so it can be "
                             "pointed at this directory directly via --data-dir.")
    parser.add_argument("--holdout-out", type=str, required=True,
                        help="Output path for the held-out 10%% real test set (.npz).")
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS,
                        help="Balancing target for the 90%% training pool -- kept the same as "
                             "the original full-data balancing target for comparability, even "
                             "though it now draws from a 10%% smaller raw pool per class.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_split(y_idx, test_frac, rng):
    """Per-class split, computed on RAW (pre-balancing, pre-duplication) indices."""
    train_idx, test_idx = [], []
    for c in np.unique(y_idx):
        idx_c = rng.permutation(np.where(y_idx == c)[0])
        n_test = int(round(len(idx_c) * test_frac))
        test_idx.append(idx_c[:n_test])
        train_idx.append(idx_c[n_test:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def balance(y_idx, target, rng):
    """Per-class undersample (no replacement) or oversample (with replacement) to `target`."""
    out_idx = []
    for c in np.unique(y_idx):
        idx_c = np.where(y_idx == c)[0]
        replace = len(idx_c) < target
        chosen = rng.choice(idx_c, size=target, replace=replace)
        out_idx.append(chosen)
    out_idx = np.concatenate(out_idx)
    rng.shuffle(out_idx)
    return out_idx


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.holdout_out) or ".", exist_ok=True)

    print("Loading raw data...")
    X = np.load(os.path.join(args.data_dir, "glitch_GAN_samples_scaled.npy"))
    y_onehot = np.load(os.path.join(args.data_dir, "glitch_GAN_labels.npy"))
    label_order = np.load(os.path.join(args.data_dir, "glitch_GAN_label_order.npy"), allow_pickle=True)
    y_idx = np.argmax(y_onehot, axis=1)
    print(f"  X: {X.shape}  y_onehot: {y_onehot.shape}  label_order: {list(label_order)}")

    for c, name in enumerate(label_order):
        print(f"  raw count [{name}]: {(y_idx == c).sum()}")

    print(f"\nSplitting {args.test_frac:.0%} held out per class (seed={args.seed})...")
    train_idx, test_idx = stratified_split(y_idx, args.test_frac, rng)
    overlap = np.intersect1d(train_idx, test_idx)
    assert len(overlap) == 0, f"BUG: {len(overlap)} indices in both train and holdout"
    print(f"  train pool: {len(train_idx)}  held-out: {len(test_idx)}")

    print(f"\nRebalancing training pool to {args.target_per_class}/class...")
    y_idx_train_pool = y_idx[train_idx]
    balanced_local_idx = balance(y_idx_train_pool, args.target_per_class, rng)
    balanced_global_idx = train_idx[balanced_local_idx]

    # Sanity check: rebalancing must only ever duplicate within the 90% train
    # pool, never reach into the held-out indices.
    assert np.all(np.isin(balanced_global_idx, train_idx))

    X_bal = X[balanced_global_idx]
    y_bal = y_onehot[balanced_global_idx]
    deriv_bal = np.diff(X_bal, axis=-1)

    for c, name in enumerate(label_order):
        n_unique_used = len(np.unique(balanced_global_idx[np.argmax(y_bal, axis=1) == c]))
        n_total = (np.argmax(y_bal, axis=1) == c).sum()
        print(f"  balanced [{name}]: {n_total} samples ({n_unique_used} unique, "
             f"{n_total - n_unique_used} duplicated)")

    np.save(os.path.join(args.out_dir, "glitch_GAN_samples_scaled_balanced.npy"), X_bal)
    np.save(os.path.join(args.out_dir, "glitch_GAN_labels_balanced.npy"), y_bal)
    np.save(os.path.join(args.out_dir, "glitch_GAN_deriv_samples_balanced.npy"), deriv_bal)
    np.save(os.path.join(args.out_dir, "glitch_GAN_label_order.npy"), label_order)
    print(f"\nSaved 90% rebalanced training pool to {args.out_dir}")

    X_holdout = X[test_idx]
    y_holdout = y_onehot[test_idx]
    np.savez(args.holdout_out, X=X_holdout, y=y_holdout, label_order=label_order)
    print(f"Saved held-out real test set to {args.holdout_out}  "
         f"(X: {X_holdout.shape}, never seen by the 90% model)")


if __name__ == "__main__":
    main()
