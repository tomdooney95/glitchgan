"""Unit tests for SNR scaling — no model weights required."""

import numpy as np
import pytest

from glitchgan.glitchgan import GlitchGAN

SRATE = 4096
N = 8192  # one standard signal length


def _compute_snr(signal: np.ndarray, srate: int = SRATE) -> float:
    """Recompute the optimal SNR of a signal in the whitened frame."""
    df = srate / signal.shape[-1]
    sig_fd = np.fft.rfft(signal, axis=-1) / srate
    sigma_sq = 4.0 * df * np.sum(np.abs(sig_fd) ** 2, axis=-1)
    return float(np.sqrt(sigma_sq))


def _random_signal(n_signals: int = 1, length: int = N) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_signals, length)).astype(np.float32)


class TestScaleSNR:
    def test_single_signal_reaches_target_snr(self):
        sig = _random_signal(1)
        target = 50.0
        scaled = GlitchGAN._scale_snr(sig, snr=target)
        assert abs(_compute_snr(scaled[0]) - target) < 0.01

    def test_batch_each_reaches_target_snr(self):
        sigs = _random_signal(8)
        target = 30.0
        scaled = GlitchGAN._scale_snr(sigs, snr=target)
        for i in range(len(scaled)):
            assert abs(_compute_snr(scaled[i]) - target) < 0.01

    def test_output_shape_preserved(self):
        sigs = _random_signal(5)
        scaled = GlitchGAN._scale_snr(sigs, snr=100.0)
        assert scaled.shape == sigs.shape

    def test_output_is_float32(self):
        sigs = _random_signal(3)
        scaled = GlitchGAN._scale_snr(sigs, snr=50.0)
        assert scaled.dtype == np.float32

    def test_different_snr_values(self):
        sig = _random_signal(1)
        for target in [10.0, 50.0, 100.0, 200.0]:
            scaled = GlitchGAN._scale_snr(sig, snr=target)
            assert abs(_compute_snr(scaled[0]) - target) < 0.1

    def test_no_nan_or_inf(self):
        sigs = _random_signal(10)
        scaled = GlitchGAN._scale_snr(sigs, snr=50.0)
        assert np.all(np.isfinite(scaled))
