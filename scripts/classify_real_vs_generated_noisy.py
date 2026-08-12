"""
Real-vs-generated discriminability test under realistic detector noise --
same classifier/split/evaluation logic as classify_real_vs_generated.py, but
operating on the noise-injected dataset produced by build_injected_dataset.py
(both domains scaled to the same per-class mean SNR, injected into
independent bilby aLIGO noise realizations) instead of the clean, noise-free
whitened waveforms.

Usage:
    python scripts/build_injected_dataset.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \\
        --out-npz injected_data/injected_real_vs_fake.npz

    python scripts/classify_real_vs_generated_noisy.py \\
        --injected-npz injected_data/injected_real_vs_fake.npz \\
        --out-dir real_vs_fake_results_noisy
"""

import argparse
import os

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from classify_real_vs_generated import build_classifier, stratified_train_val_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate the real-vs-generated CNN on the noise-injected dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--injected-npz", type=str, required=True,
                        help="Output of build_injected_dataset.py's --out-npz.")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="real_vs_fake_results_noisy")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import keras
    keras.utils.set_random_seed(args.seed)
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    rng = np.random.default_rng(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Injected dataset: {args.injected_npz}")

    d = np.load(args.injected_npz, allow_pickle=True)
    X_real, y_real_idx = d["X_real"], d["y_real_idx"]
    X_fake, y_fake_idx = d["X_fake"], d["y_fake_idx"]
    label_order = list(d["label_order"])
    print(f"Loaded: {X_real.shape[0]} real, {X_fake.shape[0]} fake "
         f"(ifo={d['ifo']}, generator={d['generator_checkpoint']})")

    X_all = np.concatenate([X_real, X_fake]).astype(np.float32)
    y_all = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_fake))]).astype(np.float32)
    class_idx_all = np.concatenate([y_real_idx, y_fake_idx]).astype(int)

    group_keys = class_idx_all * 2 + y_all.astype(int)
    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        group_keys, args.train_frac, args.val_frac, rng)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    print(f"\nSplit (stratified by class + domain): "
         f"{len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

    print("\nTraining real-vs-fake classifier on noise-injected data...")
    clf = build_classifier(input_length=X_all.shape[-1])
    clf.summary()
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True)
    history = clf.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs,
                      batch_size=args.batch_size, callbacks=[early_stop], verbose=2)
    n_ran = len(history.history["loss"])
    if n_ran < args.epochs:
        print(f"Early stopping triggered after {n_ran} epochs "
             f"(best weights restored, patience={args.patience}).")

    print("\nEvaluating on held-out classifier test split...")
    y_pred_prob = clf.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    correct = int((y_pred == y_test).sum())
    n = len(y_test)
    acc = correct / n
    lo, hi = proportion_confint(correct, n, alpha=0.05, method="wilson")

    print(f"\n{'=' * 60}")
    print(f"Real-vs-fake (noisy) classifier accuracy: {correct}/{n} = {acc:.3f}")
    print(f"95% Wilson CI: [{lo:.3f}, {hi:.3f}]")
    print(f"Chance level: 0.500 {'(INSIDE 95% CI)' if lo <= 0.5 <= hi else '(OUTSIDE 95% CI)'}")
    print(f"{'=' * 60}")

    np.savez(
        os.path.join(args.out_dir, "real_vs_fake_noisy_results.npz"),
        y_test=y_test, y_pred_prob=y_pred_prob, accuracy=acc,
        ci_lo=lo, ci_hi=hi, n=n, correct=correct,
        seed=args.seed, injected_npz=args.injected_npz,
    )
    print(f"\nSaved: {os.path.join(args.out_dir, 'real_vs_fake_noisy_results.npz')}")

    clf_path = os.path.join(args.out_dir, "classifier.keras")
    clf.save(clf_path)
    print(f"Saved trained classifier: {clf_path}")


if __name__ == "__main__":
    main()
