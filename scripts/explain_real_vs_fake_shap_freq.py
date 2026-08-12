"""
Frequency-band SHAP explainability for the real-vs-generated classifier
(classify_real_vs_generated.py).

Chunking directly in the time domain (the usual trick to make SHAP's
KernelExplainer tractable on a 8192-sample input) would smear out exactly
the kind of fine-grained/high-frequency structure we might expect a GAN to
leave as a detectable artifact. This script chunks in the FREQUENCY domain
instead: each glitch's FFT magnitude spectrum is grouped into a modest
number of frequency bands (superfeatures), and SHAP's KernelExplainer
attributes the classifier's prediction to each band by masking -- for a
given coalition of "present" bands, we keep the explained example's own
Fourier coefficients in those bands and substitute a zero-signal baseline
elsewhere, then inverse-FFT back to a waveform before running the trained
classifier. This is the standard SHAP superpixel-masking pattern (as used
for image explainers), applied to frequency bands rather than pixel
regions, so dimensionality reduction happens along an axis we don't care
about losing (fine time-localization within a band) rather than the one we
do (which frequency ranges drive the prediction).

Requires a classifier checkpoint saved by classify_real_vs_generated.py and
the `shap` package (pip install shap).

Usage:
    python scripts/explain_real_vs_fake_shap_freq.py \\
        --holdout-npz data_holdout90/holdout_real.npz \\
        --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \\
        --classifier-checkpoint real_vs_fake_results_epoch210/classifier.keras \\
        --out-dir shap_freq_results_epoch210
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
        description="Frequency-band SHAP attribution for the real-vs-generated classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--holdout-npz", type=str, required=True)
    parser.add_argument("--generator-checkpoint", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True,
                        help="classifier.keras saved by classify_real_vs_generated.py's "
                             "--out-dir.")
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--n-bands", type=int, default=32,
                        help="Number of frequency-band superfeatures.")
    parser.add_argument("--log-spaced", action="store_true", default=True,
                        help="Log-space the band edges (finer resolution at high frequency, "
                             "where GAN artifacts are more likely to hide). Use "
                             "--no-log-spaced for linear bands.")
    parser.add_argument("--no-log-spaced", dest="log_spaced", action="store_false")
    parser.add_argument("--n-explain", type=int, default=20,
                        help="Number of real and number of fake examples to explain "
                             "(so 2x this many total). SHAP cost scales with this.")
    parser.add_argument("--nsamples", type=str, default="auto",
                        help="KernelExplainer's nsamples (coalitions per explained instance). "
                             "'auto' uses SHAP's default heuristic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="shap_freq_results")
    return parser.parse_args()


def make_band_edges(n_freq, n_bands, log_spaced):
    if log_spaced:
        edges = np.unique(np.round(np.logspace(0, np.log10(n_freq), n_bands + 1)).astype(int))
        edges[0] = 0
        edges[-1] = n_freq
        if len(edges) - 1 < n_bands:
            print(f"WARNING: log-spacing collapsed to {len(edges) - 1} unique bands "
                 f"(requested {n_bands}) -- low-frequency bins are too few to log-space "
                 f"further. Consider --no-log-spaced or fewer --n-bands.")
    else:
        edges = np.linspace(0, n_freq, n_bands + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def make_predict_fn(clf, instance_fft, band_edges, n_samples):
    """Returns f(z) for SHAP's KernelExplainer: z is (n_coalitions, n_bands) binary,
    1 = keep this example's own FFT content in that band, 0 = zero it out (baseline)."""
    def predict_fn(z):
        z = np.atleast_2d(z)
        batch = np.zeros((z.shape[0], n_samples), dtype=np.float32)
        for i in range(z.shape[0]):
            fft_mod = np.zeros_like(instance_fft)
            for b, (lo, hi) in enumerate(band_edges):
                if z[i, b] == 1:
                    fft_mod[lo:hi] = instance_fft[lo:hi]
            batch[i] = np.fft.irfft(fft_mod, n=n_samples)
        return clf.predict(batch, verbose=0).ravel()
    return predict_fn


def explain_examples(clf, X, band_edges, n_samples, nsamples, label):
    import shap

    n_bands = len(band_edges)
    background = np.zeros((1, n_bands))
    all_present = np.ones((1, n_bands))
    shap_vals = np.zeros((len(X), n_bands))
    for i, x in enumerate(X):
        fft_x = np.fft.rfft(x)
        predict_fn = make_predict_fn(clf, fft_x, band_edges, n_samples)
        explainer = shap.KernelExplainer(predict_fn, background)
        sv = explainer.shap_values(all_present, nsamples=nsamples, silent=True)
        shap_vals[i] = np.asarray(sv).ravel()
        print(f"  {label} {i + 1}/{len(X)} explained")
    return shap_vals


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
    fake_parts = []
    for c, name in enumerate(label_order):
        n = int((y_real_idx == c).sum())
        fake_parts.append(
            generate_fake_samples_for_class(generator, n, c, args.num_classes, args.noise_dim, rng)
        )
    X_fake = np.concatenate(fake_parts)
    print("Applying scale_rowwise() to generated samples (matching real data's scaling)...")
    X_fake = scale_rowwise(X_fake)

    print("\nLoading trained classifier...")
    import keras
    clf = keras.models.load_model(args.classifier_checkpoint)

    n_samples = X_real.shape[-1]
    n_freq = n_samples // 2 + 1
    band_edges = make_band_edges(n_freq, args.n_bands, args.log_spaced)
    print(f"\n{len(band_edges)} frequency bands "
         f"({'log' if args.log_spaced else 'linear'}-spaced) over {n_freq} FFT bins:")
    print(f"  {band_edges}")

    n_ex = args.n_explain
    real_idx = rng.choice(len(X_real), size=min(n_ex, len(X_real)), replace=False)
    fake_idx = rng.choice(len(X_fake), size=min(n_ex, len(X_fake)), replace=False)
    real_examples = X_real[real_idx].astype(np.float32)
    fake_examples = X_fake[fake_idx].astype(np.float32)

    nsamples = args.nsamples if args.nsamples == "auto" else int(args.nsamples)

    print(f"\nExplaining {len(real_examples)} real examples (nsamples={nsamples})...")
    real_shap = explain_examples(clf, real_examples, band_edges, n_samples, nsamples, "real")
    print(f"\nExplaining {len(fake_examples)} fake examples (nsamples={nsamples})...")
    fake_shap = explain_examples(clf, fake_examples, band_edges, n_samples, nsamples, "fake")

    np.savez(
        os.path.join(args.out_dir, "shap_frequency.npz"),
        real_examples=real_examples, real_shap=real_shap,
        fake_examples=fake_examples, fake_shap=fake_shap,
        band_edges=np.array(band_edges), n_bands=args.n_bands, log_spaced=args.log_spaced,
        seed=args.seed, generator_checkpoint=args.generator_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
    )
    print(f"\nSaved: {os.path.join(args.out_dir, 'shap_frequency.npz')}")

    band_centers = [(lo + hi) / 2 for lo, hi in band_edges]
    mean_abs_real = np.abs(real_shap).mean(axis=0)
    mean_abs_fake = np.abs(fake_shap).mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(band_edges))
    width = 0.4
    ax.bar(x - width / 2, mean_abs_real, width, label="real", color="#4C72B0")
    ax.bar(x + width / 2, mean_abs_fake, width, label="fake", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.0f}" for c in band_centers], rotation=90, fontsize=7)
    ax.set_xlabel("FFT bin (band center)")
    ax.set_ylabel("mean |SHAP value|")
    ax.set_title("Per-frequency-band SHAP attribution, real vs. fake")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "shap_frequency_bands.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    print("\nIf mean |SHAP| is concentrated in the highest-index bands (high frequency) "
         "for one or both domains, that supports the high-frequency-artifact hypothesis. "
         "Compare real vs. fake bar heights per band directly in the saved figure/array.")


if __name__ == "__main__":
    main()
