Overview
========

GlitchGAN is a generative model for synthesising realistic LIGO gravitational-wave
detector glitches directly in the time domain.

The key idea
------------

Instrumental glitches — short-duration noise transients — are a major challenge for
gravitational-wave astronomy. They can mimic or obscure real signals and are difficult
to characterise at scale. GlitchGAN addresses this by learning to **generate realistic
glitch waveforms** conditioned on a glitch class, enabling:

- Augmentation of training sets for glitch classifiers
- Injection studies for signal-vs-glitch discrimination
- Morphological interpolation between glitch classes

Architecture
------------

The model is a **cDVGAN** (class-conditional Derivative GAN) with
two Wasserstein discriminators and gradient penalty:

.. code-block:: text

    Noise z ~ N(0,I)  +  class vector c  →  Generator  →  ĝ(t)
                                                               ↓
                           Discriminator D₁  (waveform realism)
                           Discriminator D₂  (derivative realism)

The derivative discriminator ``D₂`` operates on the first difference of the waveform,
penalising unrealistic temporal structure that the standard discriminator misses.
Class conditioning is injected at every layer of the generator via a class vector
``c ∈ Δ⁶`` (probability simplex over 7 classes), allowing continuous interpolation
between glitch morphologies.

Supported glitch classes
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - ``Blip``
     - Short broadband transient, < 0.1 s, 100–2000 Hz
   * - ``Fast_Scattering``
     - Periodic arches from optical path-length modulation
   * - ``Koi_Fish``
     - Long low-frequency transient with frequency evolution
   * - ``Low_Frequency_Burst``
     - Broadband burst concentrated below 100 Hz
   * - ``Scattered_Light``
     - Repeating low-frequency arches from mirror motion
   * - ``Tomte``
     - Short, loud transient with characteristic frequency evolution
   * - ``Whistle``
     - Narrowband frequency-sweeping transient

Signal specifications
---------------------

- **Sample rate**: 4096 Hz
- **Duration**: 2 seconds (8192 samples)
- **Domain**: whitened time domain
- **Pretrained epoch**: 210

Citation
--------

If you use GlitchGAN in your work, please cite:

.. code-block:: bibtex

   @article{Dooney2026GlitchGAN,
     title   = {Realistic Time-Domain Synthesis of Gravitational-Wave Detector
                Glitches using Class-Conditional Derivative Generative Adversarial Networks},
     author  = {Dooney, Tom and de Boer, Mees and Narola, Harsh and Lopez, Melissa
                and Bromuri, Stefano and Tan, Daniel Stanley and Van Den Broeck, Chris},
     journal = {Physical Review D},
     year    = {2026},
   }
