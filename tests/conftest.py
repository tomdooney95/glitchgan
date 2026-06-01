"""Shared pytest fixtures."""

from pathlib import Path

import pytest

# Path to the pretrained weights bundled in the repo
WEIGHTS_PATH = (
    Path(__file__).parent.parent
    / "weights" / "tensorflow" / "generator_210_keras3.keras"
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires the pretrained Keras weights file (slow)",
    )


@pytest.fixture(scope="session")
def model():
    """Load GlitchGAN once for the whole test session."""
    from glitchgan import GlitchGAN

    if not WEIGHTS_PATH.exists():
        pytest.skip(f"Weights file not found: {WEIGHTS_PATH}")
    return GlitchGAN.from_pretrained(str(WEIGHTS_PATH))
