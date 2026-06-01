"""User-facing GlitchGAN inference class.

Wraps the trained cDVGAN generator to provide a simple API for generating
synthetic LIGO glitches without requiring knowledge of the underlying
TensorFlow/Keras internals.
"""

from __future__ import annotations

import numpy as np

__all__ = ["GlitchGAN", "scale_for_injection"]


class GlitchGAN:
    """Generate synthetic LIGO gravitational-wave detector glitches.

    Load a pretrained generator weight file and call :meth:`generate` to
    produce time-domain glitch waveforms for any of the seven supported classes.

    Parameters
    ----------
    generator : keras.Model
        Pretrained cDVGAN generator.  Construct with :meth:`from_pretrained`
        rather than instantiating directly.

    Examples
    --------
    >>> model = GlitchGAN.from_pretrained("generator_210_keras3.keras")
    >>> blips = model.generate("Blip", n=10, snr=50)
    >>> blips.shape
    (10, 8192)
    """

    CLASSES: list[str] = [
        "Blip",
        "Fast_Scattering",
        "Koi_Fish",
        "Low_Frequency_Burst",
        "Scattered_Light",
        "Tomte",
        "Whistle",
    ]

    SAMPLE_RATE: int = 4096   # Hz
    SIGNAL_LENGTH: int = 8192  # samples — 2 s at 4096 Hz
    NOISE_DIM: int = 100

    def __init__(self, generator) -> None:
        self._generator = generator

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    #: Default Hugging Face Hub repository.
    HF_REPO_ID: str = "tomdooney/glitchgan"
    #: Filename of the generator weights inside the Hub repo.
    HF_FILENAME: str = "generator_210_keras3.keras"

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
    ) -> "GlitchGAN":
        """Load a pretrained GlitchGAN generator.

        Call with no arguments to download the default pretrained weights from
        Hugging Face Hub.  Pass ``path`` for a local ``.keras`` file, or
        ``repo_id`` to use a different Hub repository.

        Parameters
        ----------
        path : str or None
            Path to a local Keras 3 generator ``.keras`` file.  If given,
            all Hub arguments are ignored.
        repo_id : str or None
            Hugging Face Hub repository in ``"owner/name"`` format.
            Defaults to :attr:`HF_REPO_ID`.
        filename : str or None
            Filename inside the Hub repo.  Defaults to :attr:`HF_FILENAME`.
        revision : str or None
            Hub revision (branch, tag, or commit hash).  ``None`` uses the
            default branch.

        Returns
        -------
        GlitchGAN

        Examples
        --------
        Download the default pretrained weights:

        >>> model = GlitchGAN.from_pretrained()

        Load from a local file:

        >>> model = GlitchGAN.from_pretrained("generator_210_keras3.keras")
        """
        import keras
        from glitchgan.tf.model_components import ArgmaxLayer, ReduceSumDotLayer

        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=repo_id or cls.HF_REPO_ID,
                filename=filename or cls.HF_FILENAME,
                revision=revision,
            )

        generator = keras.models.load_model(
            path,
            custom_objects={
                "ArgmaxLayer": ArgmaxLayer,
                "ReduceSumDotLayer": ReduceSumDotLayer,
            },
        )
        return cls(generator)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        glitch_class: str | int | list,
        n: int = 1,
        snr: float | None = None,
    ) -> np.ndarray:
        """Generate synthetic glitch waveforms for a given class.

        Parameters
        ----------
        glitch_class : str, int, or list of str/int
            Glitch class to generate.  Pass a string (e.g. ``"Blip"``) or
            integer index.  A list of length ``n`` assigns a different class
            to each sample; a scalar is broadcast across all ``n`` samples.
        n : int
            Number of waveforms to generate.
        snr : float or None
            If given, rescale each waveform to this optimal SNR in the
            whitened frame.  If ``None``, the raw normalized generator output
            is returned.

        Returns
        -------
        numpy.ndarray, shape ``(n, 8192)``
            Time-domain glitch waveforms at 4096 Hz (2-second segments).
        """
        import tensorflow as tf

        class_tensor = self._build_class_tensor(glitch_class, n)
        noise = tf.random.normal((n, self.NOISE_DIM))
        signals = self._generator([noise, class_tensor], training=False).numpy()

        if snr is not None:
            signals = self._scale_snr(signals, snr)

        return signals

    def interpolate(
        self,
        class_weights: np.ndarray | list,
        n: int = 1,
        snr: float | None = None,
    ) -> np.ndarray:
        """Generate glitches from an explicit class-conditioning vector.

        Allows generating hybrid or transitional morphologies by specifying
        weights across multiple classes (they need not sum to 1).

        Parameters
        ----------
        class_weights : array-like, shape ``(7,)`` or ``(n, 7)``
            Class-conditioning vector(s).  A 1-D input is broadcast across
            all ``n`` samples.
        n : int
            Number of waveforms to generate.  Ignored when ``class_weights``
            has shape ``(n, 7)``.
        snr : float or None
            Target SNR — same as in :meth:`generate`.

        Returns
        -------
        numpy.ndarray, shape ``(n, 8192)``

        Examples
        --------
        Generate a glitch halfway between Blip and Tomte:

        >>> weights = [0.5, 0, 0, 0, 0, 0.5, 0]   # Blip=0, Tomte=5
        >>> hybrids = model.interpolate(weights, n=5)
        """
        import tensorflow as tf

        weights = np.asarray(class_weights, dtype=np.float32)
        if weights.ndim == 1:
            weights = np.tile(weights, (n, 1))
        else:
            n = len(weights)

        if weights.shape[1] != len(self.CLASSES):
            raise ValueError(
                f"class_weights must have {len(self.CLASSES)} elements per row, "
                f"got {weights.shape[1]}"
            )

        class_tensor = tf.cast(weights, tf.float32)
        noise = tf.random.normal((n, self.NOISE_DIM))
        signals = self._generator([noise, class_tensor], training=False).numpy()

        if snr is not None:
            signals = self._scale_snr(signals, snr)

        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_class_tensor(self, glitch_class, n):
        """Return a float32 tensor of shape (n, num_classes) for the request."""
        import tensorflow as tf

        num_classes = len(self.CLASSES)

        if isinstance(glitch_class, (list, np.ndarray)):
            indices = [self._resolve_single(c) for c in glitch_class]
            if len(indices) == 1:
                indices = indices * n
            elif len(indices) != n:
                raise ValueError(
                    f"Length of glitch_class list ({len(indices)}) must be 1 or n ({n})."
                )
        else:
            idx = self._resolve_single(glitch_class)
            indices = [idx] * n

        one_hot = np.eye(num_classes, dtype=np.float32)[indices]
        return tf.cast(one_hot, tf.float32)

    def _resolve_single(self, glitch_class) -> int:
        """Resolve a string class name or integer index to an integer."""
        if isinstance(glitch_class, str):
            if glitch_class not in self.CLASSES:
                raise ValueError(
                    f"Unknown glitch class '{glitch_class}'. "
                    f"Available: {self.CLASSES}"
                )
            return self.CLASSES.index(glitch_class)
        idx = int(glitch_class)
        if not 0 <= idx < len(self.CLASSES):
            raise ValueError(
                f"Class index {idx} out of range [0, {len(self.CLASSES) - 1}]."
            )
        return idx

    @staticmethod
    def _scale_snr(
        signals: np.ndarray,
        snr: float,
        srate: int = 4096,
    ) -> np.ndarray:
        """Rescale signals to a target optimal SNR in the whitened frame."""
        signals = np.asarray(signals, dtype=np.float64)
        df = srate / signals.shape[-1]
        sig_fd = np.fft.rfft(signals, axis=-1) / srate
        sigma_sq = 4.0 * df * np.sum(np.abs(sig_fd) ** 2, axis=-1)
        return ((signals.T) * snr / np.sqrt(sigma_sq)).T.astype(np.float32)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GlitchGAN(classes={self.CLASSES}, "
            f"sample_rate={self.SAMPLE_RATE} Hz, "
            f"signal_length={self.SIGNAL_LENGTH})"
        )


# ---------------------------------------------------------------------------
# Standalone injection utility
# ---------------------------------------------------------------------------

def scale_for_injection(
    glitch_td: np.ndarray,
    target_snr: float,
    psd: np.ndarray,
    sample_rate: float = 4096.0,
) -> np.ndarray:
    """Scale a time-domain glitch to a target matched-filter SNR.

    Uses the PSD-weighted inner product, matching the convention of
    bilby's ``Interferometer.inject_glitch()``::

        snr² = (4 / T) · Σ |h(f)|² / Sₙ(f)

    where ``h(f) = FFT(h(t)) / sample_rate`` (strain/Hz) and ``Sₙ(f)`` is
    the one-sided power spectral density of the detector noise.

    Unlike :meth:`GlitchGAN._scale_snr`, which assumes white (flat) noise,
    this correctly weights each frequency bin by the detector noise level.
    The two are equivalent only when ``psd`` is uniform.

    Parameters
    ----------
    glitch_td : array-like, shape (N,)
        Time-domain glitch waveform (single signal, not a batch).
    target_snr : float
        Desired optimal matched-filter SNR.
    psd : array-like, shape (N//2 + 1,)
        One-sided PSD ``Sₙ(f)`` at the relevant detector, evaluated at the
        frequency bins ``np.fft.rfftfreq(N, d=1/sample_rate)``.
        Obtain this from a bilby ``Interferometer`` via
        ``ifo.power_spectral_density_array``, or from gwpy via
        ``TimeSeries.psd()``.
    sample_rate : float
        Sample rate in Hz.  Must match the rate at which ``glitch_td``
        was generated (default 4096 Hz for GlitchGAN output).

    Returns
    -------
    numpy.ndarray, shape (N,)
        Rescaled time-domain glitch, same length as input.

    Examples
    --------
    Using a bilby interferometer:

    >>> import bilby, numpy as np
    >>> from glitchgan import GlitchGAN, scale_for_injection
    >>>
    >>> model = GlitchGAN.from_pretrained()
    >>> glitch = model.generate("Blip", n=1)[0]        # raw, normalized
    >>>
    >>> ifo = bilby.gw.detector.get_empty_interferometer("H1")
    >>> ifo.set_strain_data_from_power_spectral_density(
    ...     sampling_frequency=4096, duration=4.0, start_time=0.0)
    >>>
    >>> scaled = scale_for_injection(
    ...     glitch, target_snr=20.0,
    ...     psd=ifo.power_spectral_density_array,
    ...     sample_rate=4096.0)
    """
    glitch_td = np.asarray(glitch_td, dtype=np.float64).ravel()
    psd = np.asarray(psd, dtype=np.float64)

    n = len(glitch_td)
    duration = n / sample_rate

    # h(f) in strain/Hz — bilby's nfft convention: FFT(h) / sample_rate
    glitch_fd = np.fft.rfft(glitch_td) / sample_rate

    if len(glitch_fd) != len(psd):
        raise ValueError(
            f"PSD length ({len(psd)}) does not match FFT output length "
            f"({len(glitch_fd)}) for a signal of length {n}. "
            "Make sure the PSD covers the same frequency bins as "
            "np.fft.rfftfreq(len(glitch_td), d=1/sample_rate)."
        )

    # PSD-weighted optimal SNR² = (4/T) · Σ |h(f)|² / Sₙ(f)
    snr_sq = np.real(4.0 / duration * np.sum(np.abs(glitch_fd) ** 2 / psd))

    return (glitch_td * target_snr / np.sqrt(snr_sq)).astype(np.float32)
