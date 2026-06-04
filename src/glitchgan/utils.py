"""Training utilities for cDVGAN.

Includes:
- GlitchDataset         — PyTorch Dataset for GAN training data
- train_gan()           — main training loop
- generate_examples()   — vertex / simplex / uniform class sampling
- plot_losses()         — loss curve plotting
- plot_q_transform()    — Q-transform plot via gwpy
- save_checkpoint()     — save model state dicts
- load_checkpoint()     — restore model state dicts
- whitened_snr_scaling() — scale a signal to a target SNR in the whitened frame
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

# torch is only needed for the PyTorch training utilities (GlitchDataset,
# checkpointing, generate_with_noise). Import lazily so that the inference
# and plotting utilities work without torch installed.
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch(fn_name):
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"{fn_name} requires PyTorch. Install it with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GlitchDataset(Dataset):
    """Dataset for cWGAN / cDVGAN training.

    Parameters
    ----------
    signals : np.ndarray (N, L)
        Raw glitch time series.
    class_array : np.ndarray (N, num_classes)
        One-hot class labels.
    derivs : np.ndarray (N, L-1) or None
        First derivatives (required for cDVGAN / cDVGAN2).
    derivs2 : np.ndarray (N, L-2) or None
        Second derivatives (required for cDVGAN2 only).
    """

    def __init__(self, signals, class_array, derivs=None, derivs2=None):
        _require_torch("GlitchDataset")
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.classes = torch.tensor(class_array, dtype=torch.float32)
        self.derivs = torch.tensor(derivs, dtype=torch.float32) if derivs is not None else None
        self.derivs2 = torch.tensor(derivs2, dtype=torch.float32) if derivs2 is not None else None

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        out = [self.signals[idx], self.classes[idx]]
        if self.derivs is not None:
            out.append(self.derivs[idx])
        if self.derivs2 is not None:
            out.append(self.derivs2[idx])
        return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gan(gan, dataset, epochs=500, batch_size=64, save_every=25,
              monitor_every=1, output_dir="GAN_outputs", variant="cDVGAN",
              noise_dim=100, num_classes=7, start_epoch=1):
    """Train a GAN model.

    Parameters
    ----------
    gan : cWGAN | cDVGAN | cDVGAN2
        A GAN instance from glitchgan.gan_models.
    dataset : GlitchDataset
    epochs : int
        Total number of epochs to reach (not additional epochs to run).
    batch_size : int
    save_every : int
        Save full checkpoint and multi-sample example plots every N epochs.
    monitor_every : int
        Save a single vertex-sample monitor plot every N epochs (default 1).
    output_dir : str
        Directory to save checkpoints, loss history and example plots.
    variant : str
        One of ``"cWGAN"``, ``"cDVGAN"``, ``"cDVGAN2"``.
    noise_dim : int
    num_classes : int
    start_epoch : int
        Epoch to start from (1 for fresh training, or resumed checkpoint epoch + 1).

    Returns
    -------
    dict
        Loss history — keys are loss names, values are lists over epochs.
    """
    _require_torch("train_gan")
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Save training config
    config = dict(
        variant=variant,
        epochs=epochs,
        batch_size=batch_size,
        save_every=save_every,
        monitor_every=monitor_every,
        noise_dim=noise_dim,
        num_classes=num_classes,
        signal_length=dataset.signals.shape[-1],
        lr=gan.g_opt.param_groups[0]["lr"],
        d_steps=gan.d_steps,
        gp_weight=gan.gp_weight,
        device=str(gan.device),
    )
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load existing history if resuming
    history_path = os.path.join(output_dir, "history.json")
    if start_epoch > 1 and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        print(f"Resuming from epoch {start_epoch}, loaded existing loss history.")
    else:
        history = {}

    for epoch in range(start_epoch, epochs + 1):
        epoch_losses = {}

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            if variant == "cWGAN":
                signals, classes = batch
                losses = gan.train_step(signals, classes)
            elif variant == "cDVGAN":
                signals, classes, derivs = batch
                losses = gan.train_step(signals, derivs, classes)
            elif variant == "cDVGAN2":
                signals, classes, derivs, derivs2 = batch
                losses = gan.train_step(signals, derivs, derivs2, classes)
            else:
                raise ValueError(f"Unknown variant '{variant}'")

            for k, v in losses.items():
                epoch_losses.setdefault(k, []).append(v)

        # Average losses over batches
        mean_losses = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        for k, v in mean_losses.items():
            history.setdefault(k, []).append(v)

        loss_str = "  ".join(f"{k}: {v:.4f}" for k, v in mean_losses.items())
        print(f"Epoch {epoch}/{epochs}  {loss_str}")

        if monitor_every > 0 and (epoch % monitor_every == 0 or epoch == epochs):
            _monitor_epoch(gan, noise_dim, num_classes, output_dir, epoch,
                           device=gan.device)

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(gan, variant, output_dir, epoch)
            _save_examples(gan, variant, noise_dim, num_classes, output_dir, epoch,
                           device=gan.device)

    # Save loss history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f)

    plot_losses(history, variant, output_dir)
    return history


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(gan, variant, output_dir, epoch="last"):
    """Save model and optimizer state dicts to output_dir."""
    _require_torch("save_checkpoint")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(gan.generator.state_dict(),
               os.path.join(output_dir, f"generator_{epoch}.pt"))
    torch.save(gan.g_opt.state_dict(),
               os.path.join(output_dir, f"g_opt_{epoch}.pt"))

    torch.save(gan.discriminator.state_dict(),
               os.path.join(output_dir, f"discriminator_{epoch}.pt"))
    torch.save(gan.d_opt.state_dict(),
               os.path.join(output_dir, f"d_opt_{epoch}.pt"))

    if hasattr(gan, "deriv_discriminator"):
        torch.save(gan.deriv_discriminator.state_dict(),
                   os.path.join(output_dir, f"deriv_discriminator_{epoch}.pt"))
        torch.save(gan.d2d_opt.state_dict(),
                   os.path.join(output_dir, f"d2d_opt_{epoch}.pt"))

    if hasattr(gan, "deriv2_discriminator"):
        torch.save(gan.deriv2_discriminator.state_dict(),
                   os.path.join(output_dir, f"deriv2_discriminator_{epoch}.pt"))
        torch.save(gan.d2d2_opt.state_dict(),
                   os.path.join(output_dir, f"d2d2_opt_{epoch}.pt"))


def load_checkpoint(gan, output_dir, epoch="last", device="cpu"):
    """Load model and optimizer state dicts from output_dir into a GAN instance."""
    _require_torch("load_checkpoint")
    def _load(path):
        return torch.load(path, map_location=device)

    gan.generator.load_state_dict(_load(
        os.path.join(output_dir, f"generator_{epoch}.pt")))
    gan.g_opt.load_state_dict(_load(
        os.path.join(output_dir, f"g_opt_{epoch}.pt")))

    gan.discriminator.load_state_dict(_load(
        os.path.join(output_dir, f"discriminator_{epoch}.pt")))
    gan.d_opt.load_state_dict(_load(
        os.path.join(output_dir, f"d_opt_{epoch}.pt")))

    if hasattr(gan, "deriv_discriminator"):
        gan.deriv_discriminator.load_state_dict(_load(
            os.path.join(output_dir, f"deriv_discriminator_{epoch}.pt")))
        gan.d2d_opt.load_state_dict(_load(
            os.path.join(output_dir, f"d2d_opt_{epoch}.pt")))

    if hasattr(gan, "deriv2_discriminator"):
        gan.deriv2_discriminator.load_state_dict(_load(
            os.path.join(output_dir, f"deriv2_discriminator_{epoch}.pt")))
        gan.d2d2_opt.load_state_dict(_load(
            os.path.join(output_dir, f"d2d2_opt_{epoch}.pt")))


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def generate_examples(gan, noise_dim=100, num_classes=7, num_signals=10,
                      sampling="vertex", device="cpu"):
    """Generate signals using a trained generator.

    Parameters
    ----------
    gan : GAN instance
    noise_dim : int
    num_classes : int
    num_signals : int
    sampling : str
        One of ``"vertex"``, ``"simplex"``, ``"uniform"``.

        - ``"vertex"``  — pure one-hot class vectors (hard class assignment)
        - ``"simplex"`` — random convex combinations (sum to 1)
        - ``"uniform"`` — independent uniform draws per class dimension

    device : str or torch.device

    Returns
    -------
    signals : np.ndarray (num_signals, signal_length)
    class_vectors : np.ndarray (num_signals, num_classes)
    """
    device = torch.device(device)
    gan.generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_signals, noise_dim, device=device)

        if sampling == "vertex":
            indices = np.random.randint(0, num_classes, size=num_signals)
            class_vectors = np.eye(num_classes)[indices]
        elif sampling == "simplex":
            raw = np.random.randint(0, 100, size=(num_signals, num_classes)).astype(float)
            class_vectors = raw / raw.sum(axis=1, keepdims=True)
        elif sampling == "uniform":
            class_vectors = np.random.uniform(0.0, 1.0, size=(num_signals, num_classes))
        else:
            raise ValueError(f"Unknown sampling '{sampling}'. "
                             "Choose from 'vertex', 'simplex', 'uniform'.")

        class_tensor = torch.tensor(class_vectors, dtype=torch.float32, device=device)
        signals = gan.generator(noise, class_tensor).cpu().numpy()

    gan.generator.train()
    return signals, class_vectors


def _save_examples(gan, variant, noise_dim, num_classes, output_dir, epoch, device):
    """Generate and save example plots for all three sampling methods."""
    for sampling in ("vertex", "simplex", "uniform"):
        signals, classes = generate_examples(
            gan, noise_dim=noise_dim, num_classes=num_classes,
            sampling=sampling, device=device,
        )
        path = os.path.join(output_dir, f"{sampling}_examples_epoch{epoch}.png")
        _plot_examples(signals, classes, path)


def _monitor_epoch(gan, noise_dim, num_classes, output_dir, epoch, device):
    """Generate one random vertex sample and save a monitor plot."""
    monitor_dir = os.path.join(output_dir, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)
    signals, classes = generate_examples(
        gan, noise_dim=noise_dim, num_classes=num_classes,
        num_signals=1, sampling="vertex", device=device,
    )
    class_idx = int(np.argmax(classes[0]))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signals[0])
    ax.set_title(f"Epoch {epoch} — class {class_idx}")
    ax.set_xlabel("Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(monitor_dir, f"epoch_{epoch:04d}.png"))
    plt.close(fig)


def _plot_examples(signals, classes, path, n=9):
    """Plot up to n generated signals and save to path."""
    n = min(n, len(signals))
    fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        ax.plot(signals[i])
        ax.set_title(np.round(classes[i], 2), fontsize=7)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loss plotting
# ---------------------------------------------------------------------------

def plot_losses(history, variant, output_dir):
    """Plot and save training loss curves.

    Parameters
    ----------
    history : dict
        Keys are loss names, values are lists of per-epoch values.
    variant : str
    output_dir : str
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"d_loss": "C0", "d2d_loss": "C1", "d2d2_loss": "C3",
              "g_loss": "C2", "g_loss2d": "C4", "g_loss2d2": "C5",
              "g_loss_combined": "C6"}
    for key, values in history.items():
        ax.plot(values, label=key, color=colors.get(key))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{variant} training losses")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{variant}_loss_plot.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_q_transform(
    data,
    srate=4096.0,
    crop=None,
    whiten=True,
    ax=None,
    colourbar=True,
    qrange=(4, 64),
    frange=(10, 1200),
    clim=(0, 25.5),
):
    """Plot the Q-transform of a time series using gwpy.

    Parameters
    ----------
    data : array-like
        Input time-domain data.
    srate : float
        Sample rate in Hz.
    crop : tuple or None
        ``(center_time, duration)`` in seconds. ``center_time`` is measured
        from the start of ``data`` (i.e. t0=0). If None the full segment is used.
    whiten : bool
        If True, apply gwpy's internal whitening before the Q-transform.
        Pass False when the signal is already whitened or noise-free.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. A new figure is created if None.
    colourbar : bool
        Add a colorbar to the plot.
    qrange : tuple
        (q_min, q_max) range for the Q-transform.
    frange : tuple
        (f_min, f_max) frequency range in Hz.
    clim : tuple
        (vmin, vmax) colour axis limits for normalised energy.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from gwpy.timeseries import TimeSeries as GWpyTS

    ts = GWpyTS(data, sample_rate=srate)

    q_scan = ts.q_transform(
        qrange=qrange,
        frange=frange,
        tres=0.002,
        fres=0.5,
        whiten=whiten,
    )

    if isinstance(crop, (list, tuple)):
        t_center, dur = crop
        t_center = t_center + ts.t0.value
        q_scan = q_scan.crop(t_center - dur / 2, t_center + dur / 2)
        t_end = dur
    else:
        t_end = len(data) / srate
        dur = t_end

    extent = [0, t_end, frange[0], frange[1]]
    xticklabels = np.linspace(0, t_end, 5)

    if ax is None:
        _, ax = plt.subplots(dpi=120)

    im = ax.imshow(q_scan, aspect="auto", extent=extent)
    ax.set_yscale("log", base=2)
    ax.set_xscale("linear")
    ax.set_xticks(xticklabels)
    ax.set_xticklabels([f"{v:.2g}" for v in xticklabels])
    ax.set_ylabel("Frequency (Hz)", fontsize=14)
    ax.set_xlabel("Time (s)", labelpad=0.1, fontsize=14)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=14)

    im.set_clim(*clim)
    if colourbar:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.01)
        cb.ax.tick_params(labelsize=18)
        cb.set_label("Normalised energy", fontsize=24)

    return ax


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------

def whitened_snr_scaling(glitch, snr, srate=4096):
    """Scale a glitch signal to a target SNR in the whitened frame.

    Computes the true optimal SNR of the signal via its one-sided power
    spectral density, then rescales so that the injected signal has the
    requested SNR.

    Parameters
    ----------
    glitch : array-like, shape (..., N)
        Time-domain glitch signal(s).
    snr : float or None
        Target SNR. If None the signal is returned unchanged.
    srate : int
        Sample rate in Hz (default 4096).

    Returns
    -------
    numpy.ndarray
        Rescaled glitch signal(s), same shape as input.
    """
    glitch = np.asarray(glitch)
    if snr is not None:
        df = srate / glitch.shape[-1]
        glitch_FD = np.fft.rfft(glitch, axis=-1) / srate
        true_sigma_sq = (
            4.0 * df
            * np.sum(np.multiply(np.conj(glitch_FD), glitch_FD), axis=-1).real
        )
        glitch = (glitch.T * snr / np.sqrt(true_sigma_sq)).T
    return glitch
