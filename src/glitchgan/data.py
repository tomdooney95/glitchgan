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

_HOLDOUT_FILES = {
    "holdout_real": "holdout90_holdout_real.npz",
    "generator": "holdout90_generator_final.keras",
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


def download_holdout_data(
    holdout_dir: str | Path = "data_holdout90/",
    generator_dir: str | Path = "GAN_outputs_holdout90/cDVGAN/",
    include_generator: bool = True,
    force: bool = False,
) -> dict[str, Path]:
    """Download the leak-proof held-out real-data split and the holdout-trained
    GlitchGAN generator checkpoint used for the real-vs-generated
    discriminability and data-augmentation experiments (see README).

    These come from a second GlitchGAN trained on only 90% of the real data
    (see ``scripts/prepare_holdout_split.py``), so the held-out 10% was never
    seen during that model's training. This is separate from the main
    reported model downloaded by :func:`download_data`.

    Parameters
    ----------
    holdout_dir:
        Directory to save ``holdout_real.npz`` into. Created if it does not exist.
    generator_dir:
        Directory to save the holdout-trained generator checkpoint into.
    include_generator:
        If ``False``, only download the held-out real data, not the generator.
    force:
        If ``True``, re-download files even if they already exist locally.

    Returns
    -------
    dict[str, Path]
        Mapping of ``{"holdout_real"}`` (and ``"generator"`` if requested) to
        their local file paths.

    Examples
    --------
    >>> from glitchgan import download_holdout_data
    >>> paths = download_holdout_data()
    >>> import numpy as np
    >>> d = np.load(paths["holdout_real"])
    >>> X_real = d["X"]
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download data. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    dest_dirs = {"holdout_real": Path(holdout_dir), "generator": Path(generator_dir)}
    local_names = {"holdout_real": "holdout_real.npz", "generator": "generator_final.keras"}

    keys = ["holdout_real"]
    if include_generator:
        keys.append("generator")

    paths: dict[str, Path] = {}
    for key in keys:
        filename = _HOLDOUT_FILES[key]
        dest_dir = dest_dirs[key]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / local_names[key]
        if dest.exists() and not force:
            print(f"  {dest} already exists, skipping.")
            paths[key] = dest
            continue
        print(f"  Downloading {filename} ...")
        local = Path(hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset"))
        dest.write_bytes(local.read_bytes())
        paths[key] = dest
        print(f"  Saved to {paths[key]}")

    return paths
