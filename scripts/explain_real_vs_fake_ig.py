"""
Integrated Gradients explainability for the real-vs-generated classifier
(classify_real_vs_generated.py), computed at full raw-sample resolution --
one attribution value per one of the 8192 input time samples, with no
chunking/binning. This preserves whatever fine-grained (including
high-frequency) structure the classifier is actually keying on, which a
chunked/superfeature SHAP explainer (KernelExplainer et al.) would smear out.

Requires a classifier checkpoint saved by classify_real_vs_generated.py
(clf.save(...) -> classifier.keras in its --out-dir).

Usage:
    python scripts/explain_real_vs_fake_ig.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \\
        --classifier-checkpoint real_vs_fake_results_epoch210/classifier.keras \\
        --out-dir ig_results_epoch210
"""

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import numpy as np

from classify_real_vs_generated import generate_fake_samples_for_class, load_generator
from prepare_holdout_split import scale_rowwise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-resolution Integrated Gradients attribution for the "
                     "real-vs-generated classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--holdout-npz", type=str, required=True)
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True,
                        help="classifier.keras saved by classify_real_vs_generated.py's "
                             "--out-dir.")
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--n-examples", type=int, default=4,
                        help="Number of real and number of fake examples to explain/plot "
                             "(so 2x this many total).")
    parser.add_argument("--ig-steps", type=int, default=50,
                        help="Number of interpolation steps between baseline and input.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="ig_results")
    return parser.parse_args()


def integrated_gradients(model, x, baseline, steps=50):
    """Standard Integrated Gradients (Sundararajan et al. 2017) for a single
    1D input x, shape (input_length,). baseline is the same shape as x."""
    import tensorflow as tf

    alphas = np.linspace(0.0, 1.0, steps + 1).astype(np.float32)
    interpolated = baseline[None, :] + alphas[:, None] * (x[None, :] - baseline[None, :])
    interpolated_tf = tf.convert_to_tensor(interpolated, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_tf)
        preds = model(interpolated_tf, training=False)

    grads = tape.gradient(preds, interpolated_tf).numpy()
    avg_grads = grads[:-1].mean(axis=0)  # left Riemann sum over the path
    return (x - baseline) * avg_grads


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Seed: {args.seed}")
    print(f"Generator checkpoint: {args.generator_checkpoint}")
    print(f"Classifier checkpoint: {args.classifier_checkpoint}")

    print("Loading held-out real data...")
    d = np.load(args.holdout_npz, allow_pickle=True)
    X_real = d["X"]
    y_real_idx = np.argmax(d["y"], axis=1)
    label_order = list(d["label_order"])

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
    print("Applying scale_rowwise() to generated samples (matching real data's scaling)...")
    X_fake = scale_rowwise(X_fake)

    print("\nLoading trained classifier...")
    import keras
    clf = keras.models.load_model(args.classifier_checkpoint)

    n_ex = args.n_examples
    real_idx = rng.choice(len(X_real), size=min(n_ex, len(X_real)), replace=False)
    fake_idx = rng.choice(len(X_fake), size=min(n_ex, len(X_fake)), replace=False)
    real_examples = X_real[real_idx].astype(np.float32)
    fake_examples = X_fake[fake_idx].astype(np.float32)

    input_length = X_real.shape[-1]
    baseline = np.zeros(input_length, dtype=np.float32)

    print(f"\nComputing Integrated Gradients ({args.ig_steps} steps) for "
         f"{len(real_examples)} real and {len(fake_examples)} fake examples...")
    real_ig = np.stack([integrated_gradients(clf, x, baseline, args.ig_steps) for x in real_examples])
    fake_ig = np.stack([integrated_gradients(clf, x, baseline, args.ig_steps) for x in fake_examples])

    np.savez(
        os.path.join(args.out_dir, "integrated_gradients.npz"),
        real_examples=real_examples, real_ig=real_ig,
        fake_examples=fake_examples, fake_ig=fake_ig,
        seed=args.seed, generator_checkpoint=args.generator_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
    )
    print(f"Saved: {os.path.join(args.out_dir, 'integrated_gradients.npz')}")

    print("Plotting waveforms colored by attribution value...")
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    n_rows = len(real_examples) + len(fake_examples)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.0 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    # Shared color scale across all panels so blue/red intensity is directly
    # comparable between real and fake examples, not auto-scaled per panel.
    all_ig = np.concatenate([real_ig.ravel(), fake_ig.ravel()])
    vmax = np.abs(all_ig).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = "bwr"  # blue: pushes prediction toward "real"; red: pushes toward "fake"

    def plot_colored_waveform(ax, x, ig, title):
        t = np.arange(len(x))
        points = np.array([t, x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_vals = (ig[:-1] + ig[1:]) / 2.0
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(seg_vals)
        lc.set_linewidth(1.2)
        ax.add_collection(lc)
        ax.set_xlim(t.min(), t.max())
        pad = 0.1 * (x.max() - x.min() + 1e-8)
        ax.set_ylim(x.min() - pad, x.max() + pad)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("amplitude", fontsize=8)
        return lc

    row = 0
    mappable = None
    for i in range(len(real_examples)):
        mappable = plot_colored_waveform(axes[row], real_examples[i], real_ig[i], f"REAL example {i}")
        row += 1
    for i in range(len(fake_examples)):
        mappable = plot_colored_waveform(axes[row], fake_examples[i], fake_ig[i], f"FAKE example {i}")
        row += 1

    axes[-1].set_xlabel("time sample")
    fig.colorbar(mappable, ax=axes, label="IG attribution (blue: toward real, red: toward fake)",
                fraction=0.02, pad=0.02)
    fig_path = os.path.join(args.out_dir, "ig_overlay.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    print("\nDone. real_ig/fake_ig arrays in the .npz are shape (n_examples, 8192) -- "
         "one attribution value per raw time sample, safe to FFT/inspect for "
         "high-frequency structure directly without any prior chunking.")


if __name__ == "__main__":
    main()
