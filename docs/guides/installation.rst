Installation
============

Requirements
------------

- Python 3.10–3.12 (Python 3.13+ is not yet supported by TensorFlow)
- TensorFlow ≥ 2.16
- Keras ≥ 3.0

.. note::

   If you attempt to install GlitchGAN with Python 3.13 or later, pip will
   reject it immediately with a ``Requires-Python`` error.

Basic install
-------------

.. code-block:: bash

   pip install glitchgan

Recommended: conda environment
--------------------------------

TensorFlow can be tricky to install on some platforms. A conda environment
with Python 3.11 is the most reliable setup:

.. code-block:: bash

   conda create -n glitchgan python=3.11
   conda activate glitchgan
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
