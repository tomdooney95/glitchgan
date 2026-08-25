# GlitchGAN

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://glitchgan.readthedocs.io/)

Conditional Dual-discriminator Variational GAN (cDVGAN) for synthesising LIGO gravitational-wave glitch signals. Trained on seven Gravity Spy glitch classes from the O3 observing run.

## Overview

GlitchGAN uses a Wasserstein GAN with gradient penalty (WGAN-GP) augmented by a first-derivative discriminator. The derivative discriminator encourages the generator to produce signals with realistic time-domain structure, not just realistic amplitude distributions.

**Architecture:** Generator + Discriminator + Derivative Discriminator  
**Classes:** Blip, Fast Scattering, Koi Fish, Low Frequency Burst, Scattered Light, Tomte, Whistle  
**Signal length:** 8192 samples @ 4096 Hz (~2 s)  

## Repository structure

```
glitchgan/
├── evaluation.ipynb          # UMAP + GravitySpy evaluation notebook
├── src/cdvgan/
│   ├── tf/
│   │   ├── model_components.py   # Generator / discriminator layers
│   │   ├── gan_models.py         # cWGAN, cDVGAN, cDVGAN2, GlitchGAN
│   │   ├── train.py              # Training entry point
│   │   └── utils.py              # Dataset, callbacks, checkpointing
│   └── utils.py                  # Signal processing utilities
├── weights/tensorflow/
│   └── generator_210_keras3.keras   # Trained generator (epoch 210)
├── models/                       # GravitySpy CNN weights (gitignored — see below)
├── data/                         # Training data (gitignored — see below)
└── environment.yml
```

## Setup

```bash
conda env create -f environment.yml
conda activate cdvgan
```

The environment installs TensorFlow, Keras 3, GWpy, PyCBC, umap-learn, and GravitySpy.

> **Note:** `environment.yml` targets Apple Silicon (tensorflow-macos / tensorflow-metal). On Linux/HPC replace those with `tensorflow` and remove `tensorflow-metal`.

## Data

The training dataset (DeepExtractor reconstructions of seven LIGO O3 glitch classes —
Blip, Fast Scattering, Koi Fish, Low Frequency Burst, Scattered Light, Tomte, Whistle —
35,000 samples) is hosted on HuggingFace:
[tomdooney/deepextractor-glitch-reconstructions](https://huggingface.co/datasets/tomdooney/deepextractor-glitch-reconstructions)

Download it with the built-in helper:

```python
from glitchgan import download_data

paths = download_data("data/")
```

Or to also fetch the derivative array needed for cDVGAN training:

```python
paths = download_data("data/", include_derivatives=True)
```

This places the following files in `data/`:

```
data/
├── glitch_GAN_samples_scaled_balanced.npy   # (35000, 8192) whitened waveforms
├── glitch_GAN_labels_balanced.npy           # (35000, 7)    one-hot class labels
├── glitch_GAN_label_order.npy               # (7,)          class name order
└── glitch_GAN_deriv_samples_balanced.npy    # (35000, 8191) derivatives (optional)
```

### Held-out real-data split (second GlitchGAN)

The leak-proof 10% held-out real split and the holdout-trained generator checkpoint used by the
[experiments below](#additional-experiments-real-vs-generated-discriminability--data-augmentation)
are hosted in the same HuggingFace dataset repo. Download them with:

```python
from glitchgan import download_holdout_data

paths = download_holdout_data()
# paths["holdout_real"] -> data_holdout90/holdout_real.npz
# paths["generator"]    -> GAN_outputs_holdout90/cDVGAN/generator_final.keras
```

Pass `include_generator=False` to skip the checkpoint and only fetch `holdout_real.npz`.

## GravitySpy model

The GravitySpy O3 CNN (`sidd-cqg-paper-O3-model.h5`) is not included. It ships with the `gravityspy` package or can be found in a local GravitySpy clone.

1. Install GravitySpy: `pip install gravityspy`
2. Copy the model to `models/sidd-cqg-paper-O3-model.h5`
3. Set `PATH_TO_REPO` in `evaluation.ipynb` to your GravitySpy clone path

## Training

```bash
python -m cdvgan.tf.train \
    --data-dir data/ \
    --variant cDVGAN \
    --epochs 500 \
    --output-dir GAN_outputs/
```

See `src/cdvgan/tf/train.py` for all options.

## Evaluation

Open `evaluation.ipynb` and run all cells. The notebook:

1. Loads real glitch data and the trained generator
2. Visualises real vs generated waveforms
3. Embeds real and generated signals jointly in 3D UMAP space (correlation metric)
4. Injects generated signals into whitened H1 background and classifies with GravitySpy

## Additional experiments: real-vs-generated discriminability & data augmentation

Peer-review follow-up experiments testing (1) whether GlitchGAN's generated glitches are
statistically distinguishable from real ones, and (2) whether they're useful as training-data
augmentation for a downstream classifier. Both use a leak-proof held-out split of real data and
a **second GlitchGAN**, trained only on the remaining 90% pool, so the held-out real samples were
never seen during its training — intentionally separate from the main reported model (trained on
100% of the data). All scripts live in `scripts/`.

### 1. Leak-proof held-out split + the second, holdout-trained GlitchGAN

The held-out real split and the holdout-trained generator checkpoint are published — see
[Held-out real-data split](#held-out-real-data-split-second-glitchgan) above — so most users can
skip straight to step 2. To regenerate them from the raw CIT data instead:

```bash
python scripts/prepare_holdout_split.py \
    --source-npz /path/to/Compleet_set_snr15.npz \
    --out-dir data_holdout90/ \
    --holdout-out data_holdout90/holdout_real.npz
```

Splits the raw (pre-oversampling) real dataset 90/10 per class, then rebalances only the 90%
pool, writing it in the same file layout `cdvgan.tf.train` expects. Train the second GlitchGAN
on that 90% pool:

```bash
python -m cdvgan.tf.train \
    --data-dir data_holdout90/ \
    --variant cDVGAN \
    --epochs 500 \
    --output-dir GAN_outputs_holdout90/
```

### 2. Real-vs-generated discriminability (physical consistency check)

```bash
python scripts/classify_real_vs_generated.py \
    --holdout-npz data_holdout90/holdout_real.npz \
    --generator-checkpoint GAN_outputs_holdout90/cDVGAN/generator_final.keras

python scripts/classify_real_vs_generated_baseline.py \
    --holdout-npz data_holdout90/holdout_real.npz \
    --generator-checkpoint GAN_outputs_holdout90/cDVGAN/generator_final.keras
```

Trains a small standalone CNN, and a logistic-regression baseline, to tell held-out real glitches
apart from GlitchGAN-generated ones, reporting test accuracy with a 95% Wilson CI. A CI that
includes 50% supports "generated samples are statistically indistinguishable from real ones."

The same check under realistic detector noise (bilby aLIGO PSD injection):

```bash
python scripts/build_injected_dataset.py \
    --holdout-npz data_holdout90/holdout_real.npz \
    --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \
    --out-npz injected_data/injected_real_vs_fake.npz

python scripts/classify_real_vs_generated_noisy.py \
    --injected-npz injected_data/injected_real_vs_fake.npz \
    --out-dir real_vs_fake_results_noisy
```

`diagnose_real_vs_fake_stats.py`, `explain_real_vs_fake_ig.py`, and `explain_real_vs_fake_shap_freq.py`
provide diagnostics and interpretability (Integrated Gradients, SHAP) for what the classifier is
actually picking up on.

### 3. Data augmentation: does synthetic data help a downstream classifier?

```bash
python scripts/build_multiclass_dataset.py \
    --holdout-npz data_holdout90/holdout_real.npz \
    --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \
    --out-npz multiclass_data/multiclass_datasets_epoch210.npz

python scripts/classify_multiclass.py \
    --dataset-npz multiclass_data/multiclass_datasets_epoch210.npz \
    --config real_fake_augmented \
    --out-dir multiclass_results_epoch210/real_fake_augmented
```

Builds four training-set variants from the held-out real pool — real (imbalanced), real
(bootstrap-balanced), real+fake (topped up with fresh generated samples to balance), and
fake-only — all evaluated on the same fixed real val/test split so the four configs are directly
comparable. `--config` accepts `real_natural`, `real_balanced`, `real_fake_augmented`, or `fake_only`.

To check whether a result (e.g. a few-shot "augmentation rescue" effect) holds across seeds
rather than being a lucky draw from a single split:

```bash
python scripts/run_multiclass_seed_sweep.py \
    --holdout-npz data_holdout90/holdout_real.npz \
    --generator-checkpoint GAN_outputs_holdout90/cDVGAN/monitor/generator_210.keras \
    --train-samples-per-class 8 \
    --val-frac 0.2 \
    --target-per-class 5000 \
    --seeds 1,2,3,4,5 \
    --out-dir multiclass_seed_sweep_fewshot8

python scripts/summarize_all_multiclass_results.py   # aggregate across result dirs
python scripts/print_multiclass_results.py           # print a results table
```

## Citation

*(BibTeX will be added upon publication)*
