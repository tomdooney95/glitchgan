---
license: mit
tags:
  - gravitational-waves
  - time-series
  - generative-model
  - gan
  - ligo
  - detector-characterization
  - physics
library_name: keras
pipeline_tag: time-series-to-time-series
---

# GlitchGAN — Synthetic LIGO Glitch Generator

GlitchGAN is a class-conditional generative model that synthesizes realistic
gravitational-wave detector glitches directly in the time domain.
It is built on the **cDVGAN** (Conditional Derivative GAN) architecture,
which uses a first-derivative discriminator alongside the standard signal
discriminator to enforce temporal smoothness in generated waveforms.

The model is trained on high-quality glitch reconstructions produced by
[DeepExtractor](https://github.com/fbianco/deepextractor), a U-Net framework
for glitch reconstruction from LIGO strain data, covering seven common glitch
classes observed during LIGO's third observing run (O3).

## Supported glitch classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | Blip | Short, broadband transient |
| 1 | Fast\_Scattering | Arches at low frequency from scattered light |
| 2 | Koi\_Fish | Low-frequency dispersive burst |
| 3 | Low\_Frequency\_Burst | Broadband low-frequency transient |
| 4 | Scattered\_Light | Repeated arch pattern from optical scattering |
| 5 | Tomte | Short, low-frequency transient |
| 6 | Whistle | Narrow-band high-frequency chirp |

## Quick start

```bash
pip install glitchgan
```

```python
from cdvgan import GlitchGAN

# Download pretrained weights automatically (cached after first call)
model = GlitchGAN.from_pretrained()

# Generate 10 Blip glitches at SNR 50
blips = model.generate("Blip", n=10, snr=50)
# blips.shape → (10, 8192)  — 2 s at 4096 Hz

# Generate a mix of classes
glitches = model.generate(["Blip", "Tomte", "Whistle"], n=3, snr=30)

# Class interpolation — hybrid Blip/Koi Fish morphology
hybrids = model.interpolate([0.5, 0, 0.5, 0, 0, 0, 0], n=5)
```

Output waveforms are 2-second segments at 4096 Hz (8192 samples).
Pass `snr=None` (default) to get the raw normalized generator output,
or specify a target SNR to rescale to a physically meaningful amplitude.

## Model details

| Property | Value |
|----------|-------|
| Architecture | cDVGAN (conditional WGAN-GP + derivative discriminator) |
| Signal length | 8192 samples (2 s at 4096 Hz) |
| Noise dim | 100 |
| Num classes | 7 |
| Training epochs | 210 |
| Optimizer | RMSprop (lr=1e-4) |
| Gradient penalty weight | 10 |
| Framework | TensorFlow / Keras 3 |

## Training data

The model was trained on DeepExtractor reconstructions of real LIGO glitches
from O3 (2019–2020), curated using [Gravity Spy](https://www.zooniverse.org/projects/zooniverse/gravity-spy)
classifications.
Seven glitch classes were selected with balanced class sizes.
Waveforms are whitened and normalized before training; the generator
therefore learns a normalized waveform distribution and SNR scaling is
applied at inference time.

## Evaluation

Synthetic glitches were evaluated against real data using:

1. **Gravity Spy classification** — the fraction of generated samples
   correctly classified by the state-of-the-art Gravity Spy classifier.
2. **UMAP embeddings** — unsupervised 3-D projections to compare the
   morphological distributions of real and synthetic samples.

Selected Gravity Spy results at SNR 100 (100 samples per class):

| Class | Accuracy |
|-------|----------|
| Blip | 88% |
| Fast Scattering | 72% |
| Koi Fish | 20% (SNR-sensitive; 79% at SNR 150) |
| Low Frequency Burst | 100% |
| Scattered Light | 42% (often confused with Fast Scattering) |
| Tomte | 92% |
| Whistle | 94% |

UMAP embeddings show strong overlap between real and synthetic samples
for most classes. Whistle glitches show partial separation, attributed to
the model's difficulty with fine-scale high-frequency structure.

## Limitations

- **SNR sensitivity**: GlitchGAN does not model an SNR distribution.
  Generated waveforms must be manually rescaled. Classification accuracy
  for morphologically ambiguous classes (Koi Fish, Scattered Light) is
  strongly SNR-dependent.
- **Whistle class**: Synthetic Whistle glitches form a partially distinct
  cluster from real data in UMAP space, indicating incomplete capture of
  high-frequency structure.
- **Seven classes only**: The model covers the seven classes present in
  the O3 training set. It cannot generate unseen glitch morphologies.
- **LIGO-specific**: Trained on LIGO H1/L1 data. Generalization to Virgo
  or KAGRA data has not been evaluated.

## Citation

If you use GlitchGAN in your work, please cite:

```bibtex
@article{Dooney2026GlitchGAN,
  title   = {Realistic Time-Domain Synthesis of Gravitational-Wave Detector
             Glitches using Class-Conditional Derivative Generative Adversarial Networks},
  author  = {Dooney, Tom and de Boer, Mees and Narola, Harsh and Lopez, Melissa
             and Bromuri, Stefano and Tan, Daniel Stanley and Van Den Broeck, Chris},
  journal = {Physical Review D},
  year    = {2026},
}
```

## License

MIT
