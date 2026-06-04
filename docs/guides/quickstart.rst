Quickstart
==========

Generating glitches
-------------------

.. code-block:: python

   from glitchgan import GlitchGAN

   gan = GlitchGAN.from_pretrained()

   # Generate 10 Blip waveforms (returns numpy array, shape [10, 8192])
   blips = gan.generate("Blip", n=10)

   # Generate with a target matched-filter SNR
   blips_snr20 = gan.generate("Blip", n=10, snr=20)

Available class names are stored in ``GlitchGAN.CLASSES``:

.. code-block:: python

   print(GlitchGAN.CLASSES)
   # ['Blip', 'Fast_Scattering', 'Koi_Fish', 'Low_Frequency_Burst',
   #  'Scattered_Light', 'Tomte', 'Whistle']

Morphological interpolation
----------------------------

Pass a dictionary of ``{class_name: weight}`` to generate hybrid morphologies.
Weights are normalised to the unit simplex automatically.

.. code-block:: python

   # Blend 70 % Blip with 30 % Koi_Fish
   hybrid = gan.interpolate({"Blip": 0.7, "Koi_Fish": 0.3}, n=5)

Injecting into detector noise
------------------------------

Use ``scale_for_injection`` to scale a generated waveform to a target
matched-filter SNR before injecting it into real or simulated strain data.

.. code-block:: python

   from glitchgan import GlitchGAN, scale_for_injection
   from gwpy.timeseries import TimeSeries

   gan  = GlitchGAN.from_pretrained()
   koi  = gan.generate("Koi_Fish", n=1)[0]   # (8192,)

   # Fetch a background noise segment and estimate the PSD
   strain = TimeSeries.fetch_open_data("H1", 1262540000, 1262540040)
   psd    = strain.psd(4)   # 4-second segments

   scaled = scale_for_injection(koi, target_snr=20, psd=psd, sample_rate=4096)

See :doc:`../../notebooks/inject_into_detector_noise` for a complete worked example.

Notebooks
---------

* :doc:`../../notebooks/evaluation` — waveform visualisation and 3D UMAP embedding
  of real vs. generated signals

* :doc:`../../notebooks/gspy_classification` — inject generated glitches and
  classify with the GravitySpy O3 CNN

* :doc:`../../notebooks/inject_into_detector_noise` — end-to-end injection tutorial
  with ``scale_for_injection``
