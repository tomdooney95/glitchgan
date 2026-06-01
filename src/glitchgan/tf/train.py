"""Entry point for training cDVGAN models with TensorFlow/Keras.

Usage::

    python -m glitchgan.tf.train --variant cDVGAN --data-dir data/ --epochs 500

"""

import argparse
import os

# Cap TF's CPU thread pool before EagerContext is created.
# On shared HPC nodes the per-user thread limit (RLIMIT_NPROC) is typically
# ~1024, but TF 2.x defaults to one Eigen thread per physical core which can
# easily exceed that on many-core nodes.  Override with env vars so users can
# still increase them if needed (e.g. export TF_NUM_INTRAOP_THREADS=32).
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
# Disable XLA auto-JIT clustering.
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
# XLA's HloEvaluator (constant folding at compile time) creates a global Eigen
# thread pool sized to hardware_concurrency() — easily 128 threads on A100 nodes.
# This hits RLIMIT_NPROC on shared clusters.  Disable its multi-threading; only
# compile-time constant evaluation is affected, not GPU kernel execution speed.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import numpy as np
import tensorflow as tf

from glitchgan.tf.gan_models import build_gan
from glitchgan.tf.utils import train_gan


def _pick_best_gpu(gpus):
    """Return index of the GPU with the most free memory, via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free = [int(x.strip()) for x in result.stdout.strip().splitlines()]
        # Only consider GPUs visible to TF
        free = free[:len(gpus)]
        best = free.index(max(free))
        print(f"GPU free memory (MiB): {free} — selecting GPU {best}")
        return best
    except Exception as exc:
        print(f"nvidia-smi query failed ({exc}), defaulting to GPU 0")
        return 0


def _setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        _set_mg = (tf.config.set_memory_growth
                   if hasattr(tf.config, "set_memory_growth")
                   else tf.config.experimental.set_memory_growth)
        for gpu in gpus:
            _set_mg(gpu, True)
        best = _pick_best_gpu(gpus)
        tf.config.set_visible_devices(gpus[best], "GPU")
        print(f"Using GPU: {gpus[best]}")
    else:
        print("No GPU found, running on CPU.")
    print(f"TensorFlow version: {tf.__version__}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a cDVGAN model (TensorFlow backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant", type=str, default="cDVGAN",
        choices=["cWGAN", "cDVGAN", "cDVGAN2"],
        help="GAN variant to train",
    )
    parser.add_argument("--data-dir", type=str, default="data/",
                        help="Directory containing .npy data files")
    parser.add_argument("--output-dir", type=str, default="GAN_outputs_tf/",
                        help="Directory to save checkpoints and plots")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--d-steps", type=int, default=5,
                        help="Discriminator updates per generator update")
    parser.add_argument("--gp-weight", type=float, default=10.0,
                        help="Gradient penalty weight (lambda)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for RMSprop optimisers")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save model checkpoint every N epochs")
    parser.add_argument("--monitor-every", type=int, default=1,
                        help="Save a monitor plot every N epochs (0 to disable)")
    parser.add_argument("--resume-epoch", type=int, default=None,
                        help="Resume training from this checkpoint epoch")
    return parser.parse_args()


def main():
    _setup_gpu()
    args = parse_args()

    # Load data
    signals_path = os.path.join(args.data_dir, "glitch_GAN_samples_scaled_balanced.npy")
    classes_path = os.path.join(args.data_dir, "glitch_GAN_labels_balanced.npy")
    derivs_path  = os.path.join(args.data_dir, "glitch_GAN_deriv_samples_balanced.npy")

    print("Loading data...")
    signals = np.load(signals_path)
    classes = np.load(classes_path)
    derivs  = None
    derivs2 = None

    if args.variant in ("cDVGAN", "cDVGAN2"):
        derivs = np.load(derivs_path)

    if args.variant == "cDVGAN2":
        derivs2 = np.diff(derivs, axis=-1)

    signal_length = signals.shape[-1]
    print(f"Signal length: {signal_length}, N samples: {len(signals)}")

    # Build GAN
    output_dir = os.path.join(args.output_dir, args.variant)
    gan = build_gan(
        variant=args.variant,
        signal_length=signal_length,
        num_classes=args.num_classes,
        noise_dim=args.noise_dim,
        d_steps=args.d_steps,
        gp_weight=args.gp_weight,
        lr=args.lr,
    )

    # Train
    start = args.resume_epoch if args.resume_epoch is not None else 1
    print(f"Training {args.variant} from epoch {start} to {args.epochs}...")
    train_gan(
        gan=gan,
        signals=signals,
        classes=classes,
        derivs=derivs,
        derivs2=derivs2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        variant=args.variant,
        save_every=args.save_every,
        monitor_every=args.monitor_every,
        output_dir=output_dir,
        noise_dim=args.noise_dim,
        num_classes=args.num_classes,
        resume_epoch=args.resume_epoch,
    )

    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
