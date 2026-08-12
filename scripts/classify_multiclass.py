"""
7-class multi-class glitch classifier, trained on one of four dataset
variants built by build_multiclass_dataset.py (real natural/imbalanced,
real bootstrap-balanced, real+fake augmented to balance, or fake-only
balanced), always validated and tested on the SAME fixed real val/test
split for direct comparability across configurations.

Usage:
    python scripts/build_multiclass_dataset.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \\
        --out-npz multiclass_data/multiclass_datasets_epoch210.npz

    python scripts/classify_multiclass.py \\
        --dataset-npz multiclass_data/multiclass_datasets_epoch210.npz \\
        --config real_natural \\
        --out-dir multiclass_results_epoch210/real_natural
"""

import argparse
import os

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from statsmodels.stats.proportion import proportion_confint

CONFIGS = ["real_natural", "real_balanced", "real_fake_augmented", "fake_only"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate the 7-class multi-class glitch classifier on one "
                     "of the four dataset variants from build_multiclass_dataset.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-npz", type=str, required=True,
                        help="Output of build_multiclass_dataset.py's --out-npz.")
    parser.add_argument("--config", type=str, required=True, choices=CONFIGS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


def build_multiclass_classifier(input_length=8192, num_classes=7):
    """Same conv backbone as the real-vs-fake CNN (classify_real_vs_generated.py's
    build_classifier), with a softmax output over num_classes instead of a single
    sigmoid, so the two classifiers stay comparable in capacity/design philosophy."""
    import keras
    from keras import layers

    inputs = layers.Input(shape=(input_length,))
    x = layers.Reshape((input_length, 1))(inputs)
    x = layers.Conv1D(32, 16, strides=4, padding="same", activation="relu")(x)
    x = layers.Conv1D(64, 16, strides=4, padding="same", activation="relu")(x)
    x = layers.Conv1D(128, 16, strides=4, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="multiclass_glitch_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import keras
    keras.utils.set_random_seed(args.seed)
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    print(f"Seed: {args.seed}")
    print(f"Dataset: {args.dataset_npz}")
    print(f"Config: {args.config}")

    d = np.load(args.dataset_npz, allow_pickle=True)
    X_train = d[f"X_{args.config}"]
    y_train = d[f"y_{args.config}"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]
    label_order = list(d["label_order"])
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    print("\nTraining multi-class classifier...")
    clf = build_multiclass_classifier(input_length=X_train.shape[-1], num_classes=len(label_order))
    clf.summary()
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True)
    history = clf.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs,
                      batch_size=args.batch_size, callbacks=[early_stop], verbose=2)
    n_ran = len(history.history["loss"])
    if n_ran < args.epochs:
        print(f"Early stopping triggered after {n_ran} epochs "
             f"(best weights restored, patience={args.patience}).")

    print("\nEvaluating on the shared real test split...")
    y_pred_prob = clf.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    correct = int((y_pred == y_test).sum())
    n = len(y_test)
    acc = correct / n
    lo, hi = proportion_confint(correct, n, alpha=0.05, method="wilson")

    print(f"\n{'=' * 60}")
    print(f"Overall accuracy [{args.config}]: {correct}/{n} = {acc:.3f}")
    print(f"95% Wilson CI: [{lo:.3f}, {hi:.3f}]")
    print(f"{'=' * 60}")

    per_class = []
    print("\nPer-class accuracy:")
    for c, name in enumerate(label_order):
        mask = y_test == c
        n_c = int(mask.sum())
        k_c = int((y_pred[mask] == c).sum())
        acc_c = k_c / n_c if n_c > 0 else float("nan")
        if n_c > 0:
            lo_c, hi_c = proportion_confint(k_c, n_c, alpha=0.05, method="wilson")
        else:
            lo_c, hi_c = float("nan"), float("nan")
        per_class.append((name, n_c, k_c, acc_c, lo_c, hi_c))
        print(f"  [{name}] {k_c}/{n_c} = {acc_c:.3f}  95% CI [{lo_c:.3f}, {hi_c:.3f}]")

    np.savez(
        os.path.join(args.out_dir, "multiclass_results.npz"),
        config=args.config, y_test=y_test, y_pred=y_pred, y_pred_prob=y_pred_prob,
        accuracy=acc, ci_lo=lo, ci_hi=hi, n=n, correct=correct,
        per_class_names=[p[0] for p in per_class],
        per_class_n=[p[1] for p in per_class],
        per_class_correct=[p[2] for p in per_class],
        per_class_acc=[p[3] for p in per_class],
        per_class_ci_lo=[p[4] for p in per_class],
        per_class_ci_hi=[p[5] for p in per_class],
        seed=args.seed, dataset_npz=args.dataset_npz,
    )
    print(f"\nSaved: {os.path.join(args.out_dir, 'multiclass_results.npz')}")

    clf_path = os.path.join(args.out_dir, "classifier.keras")
    clf.save(clf_path)
    print(f"Saved trained classifier: {clf_path}")


if __name__ == "__main__":
    main()
