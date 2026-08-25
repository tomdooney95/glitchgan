"""cDVGAN — Conditional Dual-discriminator Variational GAN for glitch generation."""

from glitchgan.data import download_data, download_holdout_data
from glitchgan.glitchgan import GlitchGAN, scale_for_injection

__all__ = ["GlitchGAN", "scale_for_injection", "download_data", "download_holdout_data"]
