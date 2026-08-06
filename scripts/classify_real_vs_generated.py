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
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/generator_final.keras \\
        --epochs 30
"""

import argparse
import os

import numpy as np
from statsmodels.stats.proportion import proportion_confint


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
    parser.add_argument("--classifier-test-frac", type=float, default=0.2,
                        help="Fraction of the real+fake pool held out for the "
                             "CLASSIFIER's own evaluation (separate from the GAN holdout).")
    parser.add_argument("--epochs", type=int, default=30)
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


def generate_fake_samples_for_class(generator, n, class_idx, num_classes, noise_dim, rng):
    """Vertex (one-hot) class-conditioned generation for a single class,
    matching glitchgan.tf.utils.generate_examples()'s convention."""
    noise = rng.standard_normal((n, noise_dim)).astype(np.float32)
    class_vec = np.zeros((n, num_classes), dtype=np.float32)
    class_vec[:, class_idx] = 1.0
    signals = generator([noise, class_vec], training=False).numpy()
    return signals


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

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
    for c, name in enumerate(label_order):
        n = int((y_real_idx == c).sum())
        sig = generate_fake_samples_for_class(generator, n, c, args.num_classes, args.noise_dim, rng)
        fake_signals_parts.append(sig)
        print(f"    {name}: {n}")
    X_fake = np.concatenate(fake_signals_parts)
    print(f"  Generated fake: {X_fake.shape}")

    # --- Build the real(1) vs fake(0) pool ------------------------------------
    X_all = np.concatenate([X_real, X_fake]).astype(np.float32)
    y_all = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_fake))]).astype(np.float32)

    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    n_test = int(round(len(X_all) * args.classifier_test_frac))
    X_test, y_test = X_all[:n_test], y_all[:n_test]
    X_train, y_train = X_all[n_test:], y_all[n_test:]
    print(f"\nClassifier train/test split: {len(X_train)} train, {len(X_test)} test "
         f"(test_frac={args.classifier_test_frac})")

    print("\nTraining real-vs-fake classifier...")
    clf = build_classifier(input_length=X_all.shape[-1])
    clf.summary()
    clf.fit(X_train, y_train, validation_split=0.1, epochs=args.epochs,
           batch_size=args.batch_size, verbose=2)

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
    )
    print(f"\nSaved: {os.path.join(args.out_dir, 'real_vs_fake_results.npz')}")


if __name__ == "__main__":
    main()
