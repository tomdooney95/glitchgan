# PyTorch Checkpoint Weights

Copy your checkpoint files from CIT into this directory.

## Expected file format

The PyTorch trainer (`src/cdvgan/utils.py: save_checkpoint`) writes:

```
generator_<epoch>.pt          — Generator state dict
g_opt_<epoch>.pt              — Generator optimizer state
discriminator_<epoch>.pt      — Discriminator state dict
d_opt_<epoch>.pt              — Discriminator optimizer state
deriv_discriminator_<epoch>.pt     — (cDVGAN / cDVGAN2 only)
d2d_opt_<epoch>.pt                 — (cDVGAN / cDVGAN2 only)
deriv2_discriminator_<epoch>.pt    — (cDVGAN2 only)
d2d2_opt_<epoch>.pt                — (cDVGAN2 only)
config.json                   — Training hyperparameters (saved once)
history.json                  — Per-epoch loss history
```

Checkpoints were saved every 25 epochs during training on CIT.
Only `generator_<epoch>.pt` is required for inference.

## Recommended files to copy

For evaluation, copy at minimum:

- `generator_500.pt`  — end of training (500 epochs)
- `generator_<best>.pt` — whichever epoch you want to compare against the TF model at 210 epochs

Suggested rsync command from your local machine:

```bash
rsync -avz cit:/path/to/GAN_outputs/cDVGAN/generator_*.pt weights/pytorch/
rsync -avz cit:/path/to/GAN_outputs/cDVGAN/config.json weights/pytorch/
rsync -avz cit:/path/to/GAN_outputs/cDVGAN/history.json weights/pytorch/
```
