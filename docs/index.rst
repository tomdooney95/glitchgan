GlitchGAN Documentation
========================

Conditional generative model for synthesising realistic LIGO gravitational-wave
detector glitches in the time domain.

GlitchGAN uses a **class-conditional Dual-discriminator Variational GAN (cDVGAN)**
architecture — a Wasserstein GAN with gradient penalty augmented by a first-derivative
discriminator that enforces realistic temporal structure. It supports seven O3 glitch
classes and generates 2-second waveforms at 4096 Hz.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/overview
   guides/installation
   guides/quickstart
   guides/training

.. toctree::
   :maxdepth: 1
   :caption: Notebooks

   Waveforms & UMAP embedding <notebooks/evaluation.ipynb>
   GravitySpy classification <notebooks/gspy_classification.ipynb>
   Injecting into detector noise <notebooks/inject_into_detector_noise.ipynb>

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   autoapi/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
