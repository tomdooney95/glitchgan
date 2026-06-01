"""Tests for the GlitchGAN public API.

Unit tests (no model weights) run immediately.
Integration tests (marked 'integration') load the pretrained weights.
"""

import numpy as np
import pytest

from glitchgan import GlitchGAN

NUM_CLASSES = 7
SIGNAL_LENGTH = 8192


# ---------------------------------------------------------------------------
# Class-level attributes — no model needed
# ---------------------------------------------------------------------------

class TestClassAttributes:
    def test_classes_length(self):
        assert len(GlitchGAN.CLASSES) == NUM_CLASSES

    def test_classes_are_strings(self):
        assert all(isinstance(c, str) for c in GlitchGAN.CLASSES)

    def test_known_classes_present(self):
        for name in ["Blip", "Tomte", "Whistle", "Fast_Scattering",
                     "Koi_Fish", "Low_Frequency_Burst", "Scattered_Light"]:
            assert name in GlitchGAN.CLASSES

    def test_sample_rate(self):
        assert GlitchGAN.SAMPLE_RATE == 4096

    def test_signal_length(self):
        assert GlitchGAN.SIGNAL_LENGTH == SIGNAL_LENGTH


# ---------------------------------------------------------------------------
# Class resolution — no model needed
# ---------------------------------------------------------------------------

class TestResolveClass:
    @pytest.fixture(autouse=True)
    def _instance(self):
        # Create a bare instance without a real generator
        self.model = object.__new__(GlitchGAN)

    def test_resolve_by_name(self):
        assert self.model._resolve_single("Blip") == 0
        assert self.model._resolve_single("Whistle") == 6

    def test_resolve_by_index(self):
        assert self.model._resolve_single(0) == 0
        assert self.model._resolve_single(6) == 6

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown glitch class"):
            self.model._resolve_single("Earthquake")

    def test_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            self.model._resolve_single(7)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            self.model._resolve_single(-1)

    def test_build_class_tensor_shape_scalar(self):
        import tensorflow as tf
        t = self.model._build_class_tensor("Blip", n=5)
        assert t.shape == (5, NUM_CLASSES)

    def test_build_class_tensor_is_one_hot(self):
        import tensorflow as tf
        t = self.model._build_class_tensor("Blip", n=4).numpy()
        assert np.all(t.sum(axis=1) == 1.0)
        assert np.all(t[:, 0] == 1.0)  # Blip is index 0

    def test_build_class_tensor_list_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length"):
            self.model._build_class_tensor(["Blip", "Tomte"], n=5)

    def test_build_class_tensor_list_broadcast_single(self):
        import tensorflow as tf
        t = self.model._build_class_tensor(["Blip"], n=4).numpy()
        assert t.shape == (4, NUM_CLASSES)
        assert np.all(t[:, 0] == 1.0)


# ---------------------------------------------------------------------------
# Integration tests — require pretrained weights
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGenerate:
    def test_output_shape_default(self, model):
        out = model.generate("Blip", n=5)
        assert out.shape == (5, SIGNAL_LENGTH)

    def test_output_dtype(self, model):
        out = model.generate("Blip", n=3)
        assert out.dtype == np.float32

    def test_output_finite(self, model):
        out = model.generate("Blip", n=5)
        assert np.all(np.isfinite(out))

    def test_all_classes_generate(self, model):
        for cls in GlitchGAN.CLASSES:
            out = model.generate(cls, n=2)
            assert out.shape == (2, SIGNAL_LENGTH)

    def test_generate_by_index(self, model):
        out = model.generate(0, n=3)
        assert out.shape == (3, SIGNAL_LENGTH)

    def test_generate_list_of_classes(self, model):
        out = model.generate(["Blip", "Tomte", "Whistle"], n=3)
        assert out.shape == (3, SIGNAL_LENGTH)

    def test_generate_single_sample(self, model):
        out = model.generate("Blip", n=1)
        assert out.shape == (1, SIGNAL_LENGTH)

    def test_snr_scaling_applied(self, model):
        from glitchgan.glitchgan import GlitchGAN as _G
        target = 50.0
        out = model.generate("Blip", n=3, snr=target)

        def _snr(sig):
            df = GlitchGAN.SAMPLE_RATE / sig.shape[-1]
            fd = np.fft.rfft(sig) / GlitchGAN.SAMPLE_RATE
            return float(np.sqrt(4.0 * df * np.sum(np.abs(fd) ** 2)))

        for i in range(len(out)):
            assert abs(_snr(out[i]) - target) < 0.1

    def test_no_snr_returns_normalized(self, model):
        # Without SNR the generator output should be small-amplitude normalized
        out = model.generate("Blip", n=5, snr=None)
        assert out.shape == (5, SIGNAL_LENGTH)
        assert np.all(np.isfinite(out))


@pytest.mark.integration
class TestInterpolate:
    def test_output_shape(self, model):
        weights = [0.5, 0, 0, 0, 0, 0.5, 0]
        out = model.interpolate(weights, n=4)
        assert out.shape == (4, SIGNAL_LENGTH)

    def test_output_finite(self, model):
        weights = [0.5, 0, 0, 0, 0, 0.5, 0]
        out = model.interpolate(weights, n=3)
        assert np.all(np.isfinite(out))

    def test_per_sample_weights(self, model):
        weights = np.eye(NUM_CLASSES)[:3]  # shape (3, 7)
        out = model.interpolate(weights)
        assert out.shape == (3, SIGNAL_LENGTH)

    def test_wrong_num_classes_raises(self, model):
        with pytest.raises(ValueError, match="7 elements"):
            model.interpolate([0.5, 0.5], n=2)

    def test_with_snr(self, model):
        out = model.interpolate([1, 0, 0, 0, 0, 0, 0], n=2, snr=50.0)
        assert out.shape == (2, SIGNAL_LENGTH)
        assert np.all(np.isfinite(out))


@pytest.mark.integration
class TestFromPretrained:
    def test_loads_local_path(self, model):
        assert isinstance(model, GlitchGAN)

    def test_repr(self, model):
        r = repr(model)
        assert "GlitchGAN" in r
        assert "4096" in r
