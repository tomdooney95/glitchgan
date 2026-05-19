# GlitchGAN

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

The training data (~2.3 GB) is not included in this repository.

**Download from Zenodo:** *(link TBD — will be added before publication)*

Place the downloaded files in `data/`:

```
data/
├── glitch_GAN_samples_scaled_balanced.npy   # (N, 8192) float32 signals
├── glitch_GAN_labels_balanced.npy           # (N, 7) one-hot labels
└── glitch_GAN_label_order.npy               # class name ordering
```

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

## Citation

*(BibTeX will be added upon publication)*
