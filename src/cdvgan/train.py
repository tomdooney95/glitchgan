"""Entry point for training cDVGAN models.

Usage::

    python -m cdvgan.train --variant cDVGAN --data-dir data/ --epochs 500

"""

import argparse
import os

import numpy as np
import torch

from cdvgan.gan_models import build_gan
from cdvgan.utils import GlitchDataset, train_gan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a cDVGAN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant", type=str, default="cDVGAN",
        choices=["cWGAN", "cDVGAN", "cDVGAN2"],
        help="GAN variant to train",
    )
    parser.add_argument("--data-dir", type=str, default="data/",
                        help="Directory containing .npy data files")
    parser.add_argument("--output-dir", type=str, default="GAN_outputs/",
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
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save checkpoint and examples every N epochs")
    parser.add_argument("--monitor-every", type=int, default=1,
                        help="Save a monitor plot every N epochs (0 to disable)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (auto-detected if not set)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load data
    signals_path = os.path.join(args.data_dir, "glitch_GAN_samples_scaled_balanced.npy")
    classes_path = os.path.join(args.data_dir, "glitch_GAN_labels_balanced.npy")
    derivs_path  = os.path.join(args.data_dir, "glitch_GAN_deriv_samples_balanced.npy")

    print("Loading data...")
    signals = np.load(signals_path)
    classes = np.load(classes_path)

    derivs = None
    derivs2 = None

    if args.variant in ("cDVGAN", "cDVGAN2"):
        derivs = np.load(derivs_path)

    if args.variant == "cDVGAN2":
        # Second derivatives computed on the fly from the first derivatives
        derivs2 = np.diff(derivs, axis=-1)

    signal_length = signals.shape[-1]
    print(f"Signal length: {signal_length}, N samples: {len(signals)}")

    # Dataset
    dataset = GlitchDataset(signals, classes, derivs=derivs, derivs2=derivs2)

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
        device=device,
    )

    total_params = sum(p.numel() for p in gan.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")

    # Train
    print(f"Training {args.variant} for {args.epochs} epochs...")
    train_gan(
        gan=gan,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_every=args.save_every,
        monitor_every=args.monitor_every,
        output_dir=output_dir,
        variant=args.variant,
        noise_dim=args.noise_dim,
        num_classes=args.num_classes,
    )

    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
