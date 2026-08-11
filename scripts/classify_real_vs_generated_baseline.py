"""
Logistic-regression baseline for the real-vs-generated discriminability test,
run alongside the CNN in classify_real_vs_generated.py to characterize how
much model capacity is actually needed to detect a difference between real
and GlitchGAN-generated glitches -- rather than picking a classifier
architecture post-hoc based on the number it produces.

Uses the exact same data pipeline as classify_real_vs_generated.py: held-out
real samples, matched per-class generated samples, identical scale_rowwise()
normalization applied to both domains, and an identical stratified-by-class-
and-domain 70/10/20 split. Only the classifier itself changes -- an
L2-regularized LogisticRegression fit directly on the raw waveform, rather
than the 1D CNN.

If this simple linear baseline already accounts for most of the CNN's
accuracy, the real/fake difference is coarse and easily linearly separable
(e.g. the amplitude/dynamic-range spread difference found by
diagnose_real_vs_fake_stats.py). If it performs close to chance while the
CNN does not, the difference is subtler and only detectable with real
representational power.

Usage:
    python scripts/classify_real_vs_generated_baseline.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/generator_final.keras
"""

import argparse
import os

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.proportion import proportion_confint

from classify_real_vs_generated import (
    generate_fake_samples_for_class,
    load_generator,
    stratified_train_val_test_split,
)
from prepare_holdout_split import scale_rowwise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Logistic-regression baseline for the real-vs-generated "
                     "discriminability test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--holdout-npz", type=str, required=True,
                        help="Output of prepare_holdout_split.py's --holdout-out.")
    parser.add_argument("--generator-checkpoint", type=str, required=True,
                        help="Same .keras generator checkpoint used with "
                             "classify_real_vs_generated.py, for a direct comparison.")
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--C-grid", type=str, default="0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0",
                        help="Comma-separated grid of inverse L2 regularization strengths "
                             "(scikit-learn convention; smaller = stronger regularization). "
                             "The value maximizing VALIDATION accuracy is selected -- the test "
                             "set is never used for this choice.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="real_vs_fake_results_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Generator checkpoint: {args.generator_checkpoint}")

    print("Loading held-out real data...")
    d = np.load(args.holdout_npz, allow_pickle=True)
    X_real = d["X"]
    y_real_idx = np.argmax(d["y"], axis=1)
    label_order = list(d["label_order"])
    print(f"  Held-out real: {X_real.shape}")

    print("\nLoading generator and generating matching fake samples...")
    generator = load_generator(args.generator_checkpoint)
    fake_parts, fake_class_idx_parts = [], []
    for c, name in enumerate(label_order):
        n = int((y_real_idx == c).sum())
        fake_parts.append(
            generate_fake_samples_for_class(generator, n, c, args.num_classes, args.noise_dim, rng)
        )
        fake_class_idx_parts.append(np.full(n, c))
        print(f"    {name}: {n}")
    X_fake = np.concatenate(fake_parts)
    class_idx_fake = np.concatenate(fake_class_idx_parts)

    print("\nApplying scale_rowwise() to generated samples (matching real data's scaling)...")
    X_fake = scale_rowwise(X_fake)

    X_all = np.concatenate([X_real, X_fake]).astype(np.float64)
    y_all = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_fake))]).astype(np.float64)
    class_idx_all = np.concatenate([y_real_idx, class_idx_fake])

    group_keys = class_idx_all.astype(int) * 2 + y_all.astype(int)
    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        group_keys, args.train_frac, args.val_frac, rng)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    print(f"\nSplit (stratified by class + domain): "
         f"{len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

    C_grid = [float(c) for c in args.C_grid.split(",")]
    print(f"\nSweeping L2 regularization strength over C in {C_grid}, selecting by "
         f"validation accuracy (test set untouched)...")

    best_C, best_val_acc, best_clf = None, -1.0, None
    for C in C_grid:
        clf_c = LogisticRegression(C=C, max_iter=2000, random_state=args.seed)
        clf_c.fit(X_train, y_train)
        train_acc_c = clf_c.score(X_train, y_train)
        val_acc_c = clf_c.score(X_val, y_val)
        marker = ""
        if val_acc_c > best_val_acc:
            best_C, best_val_acc, best_clf = C, val_acc_c, clf_c
            marker = "  <- best so far"
        print(f"  C={C:<10g} train_acc={train_acc_c:.3f}  val_acc={val_acc_c:.3f}{marker}")

    clf = best_clf
    print(f"\nSelected C={best_C} (highest validation accuracy: {best_val_acc:.3f})")
    if best_C == max(C_grid):
        print(f"WARNING: selected C is the largest value in --C-grid -- validation accuracy "
             f"may still be rising beyond this range. Consider extending --C-grid before "
             f"trusting this as the linear model's true ceiling.")

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"Train accuracy: {train_acc:.3f}   Val accuracy: {val_acc:.3f}  "
         f"(large train/val gap would indicate overfitting)")

    y_pred = clf.predict(X_test)
    correct = int((y_pred == y_test).sum())
    n = len(y_test)
    acc = correct / n
    lo, hi = proportion_confint(correct, n, alpha=0.05, method="wilson")

    print(f"\n{'=' * 60}")
    print(f"Logistic regression baseline accuracy: {correct}/{n} = {acc:.3f}")
    print(f"95% Wilson CI: [{lo:.3f}, {hi:.3f}]")
    print(f"Chance level: 0.500 {'(INSIDE 95% CI)' if lo <= 0.5 <= hi else '(OUTSIDE 95% CI)'}")
    print(f"{'=' * 60}")
    print("\nCompare this to the CNN's accuracy from classify_real_vs_generated.py: "
         "if they're similar, the CNN isn't finding anything a simple linear model "
         "couldn't already find. If the CNN is substantially higher, the difference "
         "it's detecting is nonlinear/subtler than this baseline can capture.")

    np.savez(
        os.path.join(args.out_dir, "real_vs_fake_baseline_results.npz"),
        y_test=y_test, y_pred=y_pred, accuracy=acc, ci_lo=lo, ci_hi=hi, n=n, correct=correct,
        train_acc=train_acc, val_acc=val_acc, best_C=best_C,
        coef=clf.coef_, intercept=clf.intercept_,
        seed=args.seed, generator_checkpoint=args.generator_checkpoint,
    )
    print(f"\nSaved: {os.path.join(args.out_dir, 'real_vs_fake_baseline_results.npz')}")


if __name__ == "__main__":
    main()
