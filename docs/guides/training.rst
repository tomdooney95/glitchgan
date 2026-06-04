Training
========

Data preparation
----------------

Download the GravitySpy balanced dataset and place it in ``data/``:

.. code-block:: text

   data/
   ├── glitch_GAN_samples_scaled_balanced.npy   # (35000, 8192) whitened waveforms
   └── glitch_GAN_labels_balanced.npy            # (35000, 7)   one-hot class labels

See the ``README`` for download instructions.

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
