"""
Real-vs-generated discriminability test (physical consistency check requested
during peer review), using the held-out real data from prepare_holdout_split.py
and a GlitchGAN generator trained ONLY on the 90% training pool (so it never
saw the real samples used here).

Pipeline:
  1. Load the held-out real samples (never seen by the 90%-trained generator).
  2. Generate an equal number of fake samples per class from the 90%-trained
     generator (vertex/one-hot class conditioning, matching how GlitchGAN is
     normally sampled).
  3. Build a labeled real(1)/fake(0) dataset and split it into train/test for
     the CLASSIFIER itself -- this is a fresh, independent binary classifier,
     deliberately simpler than and separate from GlitchGAN's own
     discriminator, trained/evaluated with standard train/test discipline.
  4. Report test accuracy with a Wilson score interval (statsmodels,
     method='wilson') -- same methodology already used elsewhere in the
     paper for classification accuracy. Chance level is 50%; a CI that
     includes 50% supports the "generated samples are statistically
     indistinguishable from real ones" claim.

Usage:
    python scripts/classify_real_vs_generated.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/generator_final.keras
"""

import argparse
import os

# Shared LDG login/dev nodes (e.g. ldas-pcdev*) cap the number of threads a
# user process may spawn well below TF's default tf.data private-threadpool
# sizing (~1 thread/core), which crashes with a pthread_create() EAGAIN
# failure. Cap it conservatively before TF is ever imported; a caller that
# wants more (e.g. on a GPU compute node) can still override via env.
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from prepare_holdout_split import scale_rowwise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate a real-vs-generated classifier for the physical "
                   "consistency check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--holdout-npz", type=str, required=True,
                        help="Output of prepare_holdout_split.py's --holdout-out.")
    parser.add_argument("--generator-checkpoint", type=str, required=True,
                        help="Generator .keras checkpoint from the 90%%-holdout-trained "
                             "GlitchGAN run (NOT the main reported model) -- as saved by "
                             "glitchgan.tf.utils.save_models(), e.g. "
                             "GAN_outputs_holdout90/cDVGAN/generator_final.keras.")
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="Fraction of the real+fake pool used for the CLASSIFIER's own "
                             "training split (separate from the GAN holdout). Stratified by "
                             "class AND real/fake domain.")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of the real+fake pool used for the CLASSIFIER's own "
                             "validation split. Remainder (1 - train_frac - val_frac) is test.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max epochs; early stopping on val_loss (see --patience) will "
                             "typically halt well before this.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience on val_loss, restoring the best weights.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="real_vs_fake_results")
    return parser.parse_args()


def build_classifier(input_length=8192):
    """A small, standalone binary classifier -- deliberately simpler than and
    architecturally separate from GlitchGAN's own discriminator, so this is a
    genuinely independent check, not a re-run of the adversarial training
    objective."""
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
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="real_vs_fake_classifier")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_generator(checkpoint_path):
    """Load a .keras generator checkpoint, same convention used throughout this
    repo (evaluation.ipynb, gspy_classification.ipynb, etc.) -- NOT torch."""
    import keras
    from glitchgan.tf.model_components import ArgmaxLayer, ReduceSumDotLayer

    return keras.models.load_model(
        checkpoint_path, compile=False,
        custom_objects={"ArgmaxLayer": ArgmaxLayer, "ReduceSumDotLayer": ReduceSumDotLayer},
    )


def stratified_train_val_test_split(group_keys, train_frac, val_frac, rng):
    """Per-(class, domain) group split into train/val/test, so both the class
    imbalance in the held-out real data (277-498 per class) and the real/fake
    domain are preserved in every split, not just the pooled total."""
    train_idx, val_idx, test_idx = [], [], []
    for key in np.unique(group_keys):
        idx = rng.permutation(np.where(group_keys == key)[0])
        n = len(idx)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)


def generate_fake_samples_for_class(generator, n, class_idx, num_classes, noise_dim, rng,
                                    batch_size=500):
    """Vertex (one-hot) class-conditioned generation for a single class,
    matching glitchgan.tf.utils.generate_examples()'s convention. Generates in
    chunks of at most batch_size -- a single forward pass at large n (several
    thousand) can OOM the GPU on intermediate conv activations, even though
    the model itself is small."""
    parts = []
    remaining = n
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        noise = rng.standard_normal((this_batch, noise_dim)).astype(np.float32)
        class_vec = np.zeros((this_batch, num_classes), dtype=np.float32)
        class_vec[:, class_idx] = 1.0
        parts.append(generator([noise, class_vec], training=False).numpy())
        remaining -= this_batch
    return np.concatenate(parts, axis=0)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Seed everything before any TF/Keras op runs: numpy (used here for noise
    # vectors and the train/val/test split), and Keras's own global state
    # (weight init, dropout, and model.fit's internal batch shuffling), which
    # the numpy rng above does not control. enable_op_determinism additionally
    # forces deterministic GPU kernels (cuDNN convs are non-deterministic by
    # default for performance) at some speed cost.
    import keras
    keras.utils.set_random_seed(args.seed)
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    rng = np.random.default_rng(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Generator checkpoint: {args.generator_checkpoint}")

    print("Loading held-out real data...")
    d = np.load(args.holdout_npz, allow_pickle=True)
    X_real = d["X"]
    y_real_onehot = d["y"]
    label_order = list(d["label_order"])
    y_real_idx = np.argmax(y_real_onehot, axis=1)
    print(f"  Held-out real: {X_real.shape}")
    for c, name in enumerate(label_order):
        print(f"    {name}: {(y_real_idx == c).sum()}")

    print("\nLoading 90%-holdout-trained generator...")
    generator = load_generator(args.generator_checkpoint)

    print("Generating matching fake samples (same per-class count as held-out real)...")
    fake_signals_parts = []
    fake_class_idx_parts = []
    for c, name in enumerate(label_order):
        n = int((y_real_idx == c).sum())
        sig = generate_fake_samples_for_class(generator, n, c, args.num_classes, args.noise_dim, rng)
        fake_signals_parts.append(sig)
        fake_class_idx_parts.append(np.full(n, c))
        print(f"    {name}: {n}")
    X_fake = np.concatenate(fake_signals_parts)
    class_idx_fake = np.concatenate(fake_class_idx_parts)
    print(f"  Generated fake: {X_fake.shape}")

    # Real samples went through prepare_holdout_split.py's scale_rowwise() --
    # per-sample min-max to [-1, 1] then subtracting that sample's own mean,
    # which guarantees exactly zero row-mean. Raw generator output has no
    # equivalent enforced normalization, so without this the classifier could
    # trivially separate the two domains on row-mean/min/max alone rather than
    # on any genuine morphological difference.
    print("Applying the same row-wise min-max + mean-subtract scaling used for the "
         "real data to the generated samples...")
    X_fake = scale_rowwise(X_fake)

    # --- Build the real(1) vs fake(0) pool ------------------------------------
    X_all = np.concatenate([X_real, X_fake]).astype(np.float32)
    y_all = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_fake))]).astype(np.float32)
    class_idx_all = np.concatenate([y_real_idx, class_idx_fake])

    # Stratify by (class, domain) jointly -- class_idx in [0, num_classes) and
    # domain in {0, 1}, so this encoding is collision-free.
    group_keys = class_idx_all.astype(int) * 2 + y_all.astype(int)
    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        group_keys, args.train_frac, args.val_frac, rng)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    print(f"\nClassifier train/val/test split (stratified by class + domain): "
         f"{len(X_train)} train, {len(X_val)} val, {len(X_test)} test "
         f"(train_frac={args.train_frac}, val_frac={args.val_frac})")

    print("\nTraining real-vs-fake classifier...")
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
    else:
        print(f"Ran the full {n_ran} epochs without early stopping "
             f"-- consider raising --epochs if val_loss was still improving.")

    print("\nEvaluating on held-out classifier test split...")
    y_pred_prob = clf.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    correct = int((y_pred == y_test).sum())
    n = len(y_test)
    acc = correct / n
    lo, hi = proportion_confint(correct, n, alpha=0.05, method="wilson")

    print(f"\n{'=' * 60}")
    print(f"Real-vs-fake classifier accuracy: {correct}/{n} = {acc:.3f}")
    print(f"95% Wilson CI: [{lo:.3f}, {hi:.3f}]")
    print(f"Chance level: 0.500 {'(INSIDE 95% CI)' if lo <= 0.5 <= hi else '(OUTSIDE 95% CI)'}")
    print(f"{'=' * 60}")

    np.savez(
        os.path.join(args.out_dir, "real_vs_fake_results.npz"),
        y_test=y_test, y_pred_prob=y_pred_prob, accuracy=acc,
        ci_lo=lo, ci_hi=hi, n=n, correct=correct,
        seed=args.seed, generator_checkpoint=args.generator_checkpoint,
    )
    print(f"\nSaved: {os.path.join(args.out_dir, 'real_vs_fake_results.npz')}")

    clf_path = os.path.join(args.out_dir, "classifier.keras")
    clf.save(clf_path)
    print(f"Saved trained classifier: {clf_path}")


if __name__ == "__main__":
    main()
