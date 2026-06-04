Installation
============

Requirements
------------

- Python ≥ 3.10
- TensorFlow ≥ 2.16
- Keras ≥ 3.0

Basic install
-------------

.. code-block:: bash

   pip install glitchgan

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/tomdooney95/glitchgan.git
   cd glitchgan
   pip install -e .

Full evaluation stack (conda)
------------------------------

The evaluation notebooks additionally require GWpy, PyCBC, UMAP, and GravitySpy.
A complete conda environment is provided:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate cdvgan

Optional documentation dependencies
-------------------------------------

.. code-block:: bash

   pip install glitchgan[docs]

Pretrained weights
------------------

Pretrained generator weights are hosted on
`Hugging Face Hub <https://huggingface.co/tomdooney95/glitchgan>`_ and are
downloaded automatically on first use — no manual step required.

To download explicitly:

.. code-block:: python

   from glitchgan import GlitchGAN

   gan = GlitchGAN.from_pretrained()   # downloads and caches weights
