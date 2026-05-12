"""Training utilities for the TensorFlow cDVGAN implementation.

Includes:
- build_dataset()       — build a tf.data.Dataset from numpy arrays
- GANMonitor            — Keras callback for per-epoch signal plots
- train_gan()           — wraps model.fit() with checkpointing and history
- generate_examples()   — vertex / simplex / uniform class sampling
- save_models()         — save all model components in .keras format
- load_models()         — restore model components from .keras files
- plot_losses()         — loss curve plotting
- plot_examples()       — grid plot of generated signals
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(signals, classes, derivs=None, derivs2=None,
                  batch_size=64, shuffle_buffer=1000):
    """Build a tf.data.Dataset for GAN training.

    Parameters
    ----------
    signals : np.ndarray (N, L)
    classes : np.ndarray (N, num_classes)
    derivs : np.ndarray (N, L-1) or None
    derivs2 : np.ndarray (N, L-2) or None
    batch_size : int
    shuffle_buffer : int

    Returns
    -------
    tf.data.Dataset
        Each element is a tuple matching what the GAN's train_step expects:
        - cWGAN   : (signals, classes)
        - cDVGAN  : (signals, derivs, classes)
        - cDVGAN2 : (signals, derivs, derivs2, classes)
    """
    arrays = [signals.astype(np.float32), ]
    if derivs is not None:
        arrays.append(derivs.astype(np.float32))
    if derivs2 is not None:
        arrays.append(derivs2.astype(np.float32))
    arrays.append(classes.astype(np.float32))

    dataset = tf.data.Dataset.from_tensor_slices(tuple(arrays))
    dataset = (dataset
               .shuffle(shuffle_buffer)
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))
    return dataset


# ---------------------------------------------------------------------------
# Keras callback: per-epoch monitor
# ---------------------------------------------------------------------------

class GANMonitor(keras.callbacks.Callback):
    """Save a generated signal plot and model checkpoints during training."""

    def __init__(self, noise_dim=100, num_classes=7, output_dir="monitor",
                 save_model_every=10):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.save_model_every = save_model_every
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        idx = np.random.randint(0, self.num_classes)
        class_vec = tf.one_hot([idx], self.num_classes, on_value=1.0, off_value=0.0)
        noise = tf.random.normal((1, self.noise_dim))
        signal = self.model.generator([noise, class_vec], training=False).numpy()[0]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(signal)
        ax.set_title(f"Epoch {epoch + 1} — class {idx}")
        ax.set_xlabel("Sample")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"epoch_{epoch + 1:04d}.png"))
        plt.close(fig)

        if (epoch + 1) % self.save_model_every == 0:
            save_models(self.model, self.output_dir, epoch=(epoch + 1))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gan(gan, signals, classes, derivs=None, derivs2=None,
              epochs=500, batch_size=64, variant="cDVGAN",
              save_every=25, monitor_every=1, output_dir="GAN_outputs",
              noise_dim=100, num_classes=7, resume_epoch=None):
    """Train a TF GAN model.

    Parameters
    ----------
    gan : keras.Model  (cWGAN / cDVGAN / cDVGAN2)
    signals : np.ndarray (N, L)
    classes : np.ndarray (N, num_classes)
    derivs : np.ndarray or None
    derivs2 : np.ndarray or None
    epochs : int
    batch_size : int
    variant : str
    save_every : int
    monitor_every : int   (0 to disable per-epoch plots)
    output_dir : str
    noise_dim : int
    num_classes : int

    Returns
    -------
    dict  — loss history
    """
    os.makedirs(output_dir, exist_ok=True)

    initial_epoch = 0
    if resume_epoch is not None:
        print(f"Resuming from epoch {resume_epoch}...")
        load_models(gan, output_dir, epoch=resume_epoch)
        initial_epoch = resume_epoch

    dataset = build_dataset(signals, classes, derivs=derivs, derivs2=derivs2,
                            batch_size=batch_size)

    callbacks = []
    if monitor_every > 0:
        monitor_dir = os.path.join(output_dir, "monitor")
        callbacks.append(GANMonitor(
            noise_dim=noise_dim, num_classes=num_classes,
            output_dir=monitor_dir, save_model_every=save_every,
        ))

    # Keras model.fit() requires compile() to have been called, but our custom
    # train_step manages its own optimizers. Call compile with no args.
    gan.compile()

    history = gan.fit(dataset, epochs=epochs, initial_epoch=initial_epoch,
                      callbacks=callbacks)

    # Save final models and history
    save_models(gan, output_dir, epoch="final")

    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history_dict, f)

    plot_losses(history_dict, variant, output_dir)

    # Generate example plots
    for sampling in ("vertex", "simplex", "uniform"):
        sigs, classes_out = generate_examples(
            gan, noise_dim=noise_dim, num_classes=num_classes, sampling=sampling)
        path = os.path.join(output_dir, f"{sampling}_examples_final.png")
        plot_examples(sigs, classes_out, path)

    return history_dict


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_models(gan, output_dir, epoch="last"):
    """Save all model components in .keras format plus optimizer states."""
    os.makedirs(output_dir, exist_ok=True)

    gan.generator.save(os.path.join(output_dir, f"generator_{epoch}.keras"))
    gan.discriminator.save(os.path.join(output_dir, f"discriminator_{epoch}.keras"))
    if hasattr(gan, "deriv_discriminator"):
        gan.deriv_discriminator.save(
            os.path.join(output_dir, f"deriv_discriminator_{epoch}.keras"))
    if hasattr(gan, "deriv2_discriminator"):
        gan.deriv2_discriminator.save(
            os.path.join(output_dir, f"deriv2_discriminator_{epoch}.keras"))

    ckpt_kwargs = {"g_optimizer": gan.g_optimizer, "d_optimizer": gan.d_optimizer}
    if hasattr(gan, "d2d_optimizer"):
        ckpt_kwargs["d2d_optimizer"] = gan.d2d_optimizer
    if hasattr(gan, "d2d2_optimizer"):
        ckpt_kwargs["d2d2_optimizer"] = gan.d2d2_optimizer
    tf.train.Checkpoint(**ckpt_kwargs).write(
        os.path.join(output_dir, f"optimizers_{epoch}"))


def load_models(gan, output_dir, epoch="last"):
    """Restore model weights and optimizer states from a checkpoint."""
    gan.generator = keras.models.load_model(
        os.path.join(output_dir, f"generator_{epoch}.keras"))
    gan.discriminator = keras.models.load_model(
        os.path.join(output_dir, f"discriminator_{epoch}.keras"))
    if hasattr(gan, "deriv_discriminator"):
        gan.deriv_discriminator = keras.models.load_model(
            os.path.join(output_dir, f"deriv_discriminator_{epoch}.keras"))
    if hasattr(gan, "deriv2_discriminator"):
        gan.deriv2_discriminator = keras.models.load_model(
            os.path.join(output_dir, f"deriv2_discriminator_{epoch}.keras"))

    # Optimizer slot variables are created lazily on first apply_gradients;
    # deferred restore ensures they are populated as soon as they exist.
    ckpt_kwargs = {"g_optimizer": gan.g_optimizer, "d_optimizer": gan.d_optimizer}
    if hasattr(gan, "d2d_optimizer"):
        ckpt_kwargs["d2d_optimizer"] = gan.d2d_optimizer
    if hasattr(gan, "d2d2_optimizer"):
        ckpt_kwargs["d2d2_optimizer"] = gan.d2d2_optimizer
    tf.train.Checkpoint(**ckpt_kwargs).restore(
        os.path.join(output_dir, f"optimizers_{epoch}")).expect_partial()


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def generate_examples(gan, noise_dim=100, num_classes=7, num_signals=10,
                      sampling="vertex"):
    """Generate signals from the trained generator.

    Parameters
    ----------
    sampling : str
        One of ``"vertex"``, ``"simplex"``, ``"uniform"``.

    Returns
    -------
    signals : np.ndarray (num_signals, signal_length)
    class_vectors : np.ndarray (num_signals, num_classes)
    """
    noise = tf.random.normal((num_signals, noise_dim))

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

    class_tensor = tf.cast(class_vectors, tf.float32)
    signals = gan.generator([noise, class_tensor], training=False).numpy()
    return signals, class_vectors


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_losses(history, variant, output_dir):
    """Plot and save training loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"d_loss": "C0", "d2d_loss": "C1", "d2d2_loss": "C3",
              "g_loss": "C2", "g_loss2d": "C4", "g_loss2d2": "C5",
              "g_loss_combined": "C6"}
    for key, values in history.items():
        ax.plot(values, label=key, color=colors.get(key))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{variant} training losses (TF)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{variant}_loss_plot.png"))
    plt.close(fig)


def plot_examples(signals, classes, path, n=9):
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
