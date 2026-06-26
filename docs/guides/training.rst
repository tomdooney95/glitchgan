Training
========

Data preparation
----------------

Download the training dataset (DeepExtractor reconstructions of seven LIGO O3 glitch classes)
directly from HuggingFace using the built-in helper:

.. code-block:: python

   from glitchgan import download_data

   paths = download_data("data/")
   # paths["samples"]     → data/glitch_GAN_samples_scaled_balanced.npy
   # paths["labels"]      → data/glitch_GAN_labels_balanced.npy
   # paths["label_order"] → data/glitch_GAN_label_order.npy

To also download the time-derivative array required for cDVGAN training (~2.1 GB extra):

.. code-block:: python

   paths = download_data("data/", include_derivatives=True)
   # paths["derivatives"] → data/glitch_GAN_deriv_samples_balanced.npy

The dataset is hosted at
`tomdooney/deepextractor-glitch-reconstructions <https://huggingface.co/datasets/tomdooney/deepextractor-glitch-reconstructions>`_
on HuggingFace (35,000 samples, 7 classes, 8192 samples at 4096 Hz).

Expected directory layout after download:

.. code-block:: text

   data/
   ├── glitch_GAN_samples_scaled_balanced.npy   # (35000, 8192) whitened waveforms
   ├── glitch_GAN_labels_balanced.npy            # (35000, 7)   one-hot class labels
   ├── glitch_GAN_label_order.npy                # (7,)         class name order
   └── glitch_GAN_deriv_samples_balanced.npy     # (35000, 8191) derivatives (optional)

Training a model
----------------

.. code-block:: bash

   glitchgan-train \
       --variant cDVGAN \
       --data-dir data/ \
       --epochs 500 \
       --output-dir outputs/

Available model variants
------------------------

.. list-table::
   :header-rows: 1

   * - Variant
     - Description
   * - ``cWGAN``
     - Conditional Wasserstein GAN with gradient penalty (single discriminator)
   * - ``cDVGAN``
     - Dual-discriminator cWGAN with derivative discriminator (recommended)
   * - ``cDVGAN2``
     - Extended cDVGAN with additional second-derivative discriminator

Python API
----------

.. code-block:: python

   from glitchgan.tf import build_gan, train_gan, GlitchDataset
   import numpy as np

   X = np.load("data/glitch_GAN_samples_scaled_balanced.npy")
   y = np.load("data/glitch_GAN_labels_balanced.npy")

   dataset = GlitchDataset(X, y, batch_size=64)
   gan     = build_gan("cDVGAN", noise_dim=100, num_classes=7, signal_length=8192)

   train_gan(gan, dataset, epochs=500, checkpoint_dir="checkpoints/")

Checkpointing
-------------

Weights are saved every 10 epochs to ``checkpoint_dir/``. Training can be resumed
by pointing ``--output-dir`` at an existing checkpoint directory.

Run ``glitchgan-train --help`` for the full list of arguments.
