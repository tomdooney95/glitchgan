"""Utilities for downloading the GlitchGAN training dataset from HuggingFace."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ID = "tomdooney/deepextractor-glitch-reconstructions"

_FILES = {
    "samples": "glitch_GAN_samples_scaled_balanced.npy",
    "labels": "glitch_GAN_labels_balanced.npy",
    "label_order": "glitch_GAN_label_order.npy",
    "derivatives": "glitch_GAN_deriv_samples_balanced.npy",
}


def download_data(
    data_dir: str | Path = "data/",
    include_derivatives: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Download the GlitchGAN training dataset from HuggingFace.

    Downloads DeepExtractor reconstructions of seven LIGO glitch classes
    (35,000 samples, balanced across Blip, Fast Scattering, Koi Fish,
    Low Frequency Burst, Scattered Light, Tomte, and Whistle) from
    ``tomdooney/deepextractor-glitch-reconstructions`` on HuggingFace.

    Parameters
    ----------
    data_dir:
        Directory to save downloaded files. Created if it does not exist.
    include_derivatives:
        If ``True``, also download the first-order time-derivative array
        required for cDVGAN training (adds ~2.1 GB).
    force:
        If ``True``, re-download files even if they already exist locally.

    Returns
    -------
    dict[str, Path]
        Mapping of ``{"samples", "labels", "label_order"}`` (and
        ``"derivatives"`` if requested) to their local file paths.

    Examples
    --------
    >>> from glitchgan import download_data
    >>> paths = download_data("data/")
    >>> import numpy as np
    >>> X = np.load(paths["samples"])   # (35000, 8192)
    >>> y = np.load(paths["labels"])    # (35000, 7)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download data. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    keys = ["samples", "labels", "label_order"]
    if include_derivatives:
        keys.append("derivatives")

    paths: dict[str, Path] = {}
    for key in keys:
        filename = _FILES[key]
        dest = data_dir / filename
        if dest.exists() and not force:
            print(f"  {filename} already exists, skipping.")
            paths[key] = dest
            continue
        print(f"  Downloading {filename} ...")
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(data_dir),
        )
        paths[key] = Path(local)
        print(f"  Saved to {paths[key]}")

    return paths
