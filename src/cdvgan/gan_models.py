"""PyTorch GAN model wrappers for cDVGAN.

Includes:
- cWGAN   — conditional Wasserstein GAN with gradient penalty
- cDVGAN  — adds a derivative discriminator
- cDVGAN2 — adds first and second derivative discriminators
"""

import torch
import torch.nn as nn
import torch.optim as optim

from cdvgan.model_components import (
    Discriminator,
    DerivativeDiscriminator,
    Generator,
    SecondDerivativeDiscriminator,
)

NUM_CLASSES = 7


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _gradient_penalty(discriminator, real, fake, class_vector, device):
    """Compute the WGAN-GP gradient penalty for a given discriminator.

    Parameters
    ----------
    discriminator : nn.Module
    real : Tensor (batch, L)
    fake : Tensor (batch, L)
    class_vector : Tensor (batch, num_classes)
    device : torch.device

    Returns
    -------
    Tensor scalar
    """
    batch = real.size(0)
    alpha = torch.rand(batch, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    pred = discriminator(interpolated, class_vector)

    grads = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    norm = grads.norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


def wasserstein_discriminator_loss(real_logits, fake_logits):
    """Wasserstein loss for discriminator: fake - real."""
    return fake_logits.mean() - real_logits.mean()


def wasserstein_generator_loss(fake_logits):
    """Wasserstein loss for generator: -fake."""
    return -fake_logits.mean()


# ---------------------------------------------------------------------------
# cWGAN
# ---------------------------------------------------------------------------

class cWGAN(nn.Module):
    """Conditional Wasserstein GAN with gradient penalty.

    Parameters
    ----------
    signal_length : int
    num_classes : int
    noise_dim : int
    d_steps : int
        Number of discriminator updates per generator update.
    gp_weight : float
        Gradient penalty weight (lambda).
    lr : float
        Learning rate for RMSprop optimisers.
    device : str or torch.device
    """

    def __init__(self, signal_length=8192, num_classes=NUM_CLASSES, noise_dim=100,
                 d_steps=5, gp_weight=10.0, lr=1e-4, device="cpu"):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.device = torch.device(device)

        self.generator = Generator(noise_dim, num_classes, signal_length).to(self.device)
        self.discriminator = Discriminator(signal_length, num_classes).to(self.device)

        self.g_opt = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.d_opt = optim.RMSprop(self.discriminator.parameters(), lr=lr)

    def train_step(self, real_signals, class_vector):
        """One training step.

        Parameters
        ----------
        real_signals : Tensor (batch, signal_length)
        class_vector : Tensor (batch, num_classes)

        Returns
        -------
        dict with keys: d_loss, g_loss
        """
        real_signals = real_signals.to(self.device)
        class_vector = class_vector.to(self.device)
        batch = real_signals.size(0)

        # --- Discriminator steps ---
        for _ in range(self.d_steps):
            noise = torch.randn(batch, self.noise_dim, device=self.device)
            fake = self.generator(noise, class_vector).detach()

            real_logits = self.discriminator(real_signals, class_vector)
            fake_logits = self.discriminator(fake, class_vector)
            gp = _gradient_penalty(self.discriminator, real_signals, fake, class_vector, self.device)

            d_loss = wasserstein_discriminator_loss(real_logits, fake_logits) + self.gp_weight * gp

            self.d_opt.zero_grad()
            d_loss.backward()
            self.d_opt.step()

        # --- Generator step ---
        noise = torch.randn(batch, self.noise_dim, device=self.device)
        fake = self.generator(noise, class_vector)
        g_loss = wasserstein_generator_loss(self.discriminator(fake, class_vector))

        self.g_opt.zero_grad()
        g_loss.backward()
        self.g_opt.step()

        return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}


# ---------------------------------------------------------------------------
# cDVGAN
# ---------------------------------------------------------------------------

class cDVGAN(nn.Module):
    """Conditional Dual-discriminator Variational GAN.

    Adds a derivative discriminator that operates on the first derivative
    of signals, providing an additional regularisation signal.

    Parameters
    ----------
    signal_length : int
    num_classes : int
    noise_dim : int
    d_steps : int
    gp_weight : float
    lr : float
    device : str or torch.device
    """

    def __init__(self, signal_length=8192, num_classes=NUM_CLASSES, noise_dim=100,
                 d_steps=5, gp_weight=10.0, lr=1e-4, device="cpu"):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.device = torch.device(device)

        self.generator = Generator(noise_dim, num_classes, signal_length).to(self.device)
        self.discriminator = Discriminator(signal_length, num_classes).to(self.device)
        self.deriv_discriminator = DerivativeDiscriminator(signal_length - 1, num_classes).to(self.device)

        self.g_opt = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.d_opt = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        self.d2d_opt = optim.RMSprop(self.deriv_discriminator.parameters(), lr=lr)

    def train_step(self, real_signals, real_derivs, class_vector):
        """One training step.

        Parameters
        ----------
        real_signals : Tensor (batch, signal_length)
        real_derivs : Tensor (batch, signal_length-1)
        class_vector : Tensor (batch, num_classes)

        Returns
        -------
        dict with keys: d_loss, d2d_loss, g_loss, g_loss2d, g_loss_combined
        """
        real_signals = real_signals.to(self.device)
        real_derivs = real_derivs.to(self.device)
        class_vector = class_vector.to(self.device)
        batch = real_signals.size(0)

        # --- Discriminator steps ---
        for _ in range(self.d_steps):
            noise = torch.randn(batch, self.noise_dim, device=self.device)
            fake = self.generator(noise, class_vector).detach()
            fake_derivs = torch.diff(fake, dim=-1)

            # Signal discriminator
            real_logits = self.discriminator(real_signals, class_vector)
            fake_logits = self.discriminator(fake, class_vector)
            gp = _gradient_penalty(self.discriminator, real_signals, fake, class_vector, self.device)
            d_loss = wasserstein_discriminator_loss(real_logits, fake_logits) + self.gp_weight * gp

            self.d_opt.zero_grad()
            d_loss.backward()
            self.d_opt.step()

            # Derivative discriminator
            fake_derivs = torch.diff(self.generator(
                torch.randn(batch, self.noise_dim, device=self.device), class_vector
            ).detach(), dim=-1)

            real_logits2d = self.deriv_discriminator(real_derivs, class_vector)
            fake_logits2d = self.deriv_discriminator(fake_derivs, class_vector)
            gp2d = _gradient_penalty(self.deriv_discriminator, real_derivs, fake_derivs, class_vector, self.device)
            d2d_loss = wasserstein_discriminator_loss(real_logits2d, fake_logits2d) + self.gp_weight * gp2d

            self.d2d_opt.zero_grad()
            d2d_loss.backward()
            self.d2d_opt.step()

        # --- Generator step ---
        noise = torch.randn(batch, self.noise_dim, device=self.device)
        fake = self.generator(noise, class_vector)
        fake_derivs = torch.diff(fake, dim=-1)

        g_loss = wasserstein_generator_loss(self.discriminator(fake, class_vector))
        g_loss2d = wasserstein_generator_loss(self.deriv_discriminator(fake_derivs, class_vector))
        g_loss_combined = 0.5 * (g_loss + g_loss2d)

        self.g_opt.zero_grad()
        g_loss_combined.backward()
        self.g_opt.step()

        return {
            "d_loss": d_loss.item(),
            "d2d_loss": d2d_loss.item(),
            "g_loss": g_loss.item(),
            "g_loss2d": g_loss2d.item(),
            "g_loss_combined": g_loss_combined.item(),
        }


# ---------------------------------------------------------------------------
# cDVGAN2
# ---------------------------------------------------------------------------

class cDVGAN2(nn.Module):
    """cDVGAN extended with a second-derivative discriminator.

    Parameters
    ----------
    signal_length : int
    num_classes : int
    noise_dim : int
    d_steps : int
    gp_weight : float
    lr : float
    device : str or torch.device
    """

    def __init__(self, signal_length=8192, num_classes=NUM_CLASSES, noise_dim=100,
                 d_steps=5, gp_weight=10.0, lr=1e-4, device="cpu"):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.device = torch.device(device)

        self.generator = Generator(noise_dim, num_classes, signal_length).to(self.device)
        self.discriminator = Discriminator(signal_length, num_classes).to(self.device)
        self.deriv_discriminator = DerivativeDiscriminator(signal_length - 1, num_classes).to(self.device)
        self.deriv2_discriminator = SecondDerivativeDiscriminator(signal_length - 2, num_classes).to(self.device)

        self.g_opt = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.d_opt = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        self.d2d_opt = optim.RMSprop(self.deriv_discriminator.parameters(), lr=lr)
        self.d2d2_opt = optim.RMSprop(self.deriv2_discriminator.parameters(), lr=lr)

    def _fake_and_derivs(self, batch, class_vector):
        noise = torch.randn(batch, self.noise_dim, device=self.device)
        fake = self.generator(noise, class_vector)
        fake_d1 = torch.diff(fake, n=1, dim=-1)
        fake_d2 = torch.diff(fake, n=2, dim=-1)
        return fake, fake_d1, fake_d2

    def train_step(self, real_signals, real_derivs, real_derivs2, class_vector):
        """One training step.

        Parameters
        ----------
        real_signals : Tensor (batch, signal_length)
        real_derivs : Tensor (batch, signal_length-1)
        real_derivs2 : Tensor (batch, signal_length-2)
        class_vector : Tensor (batch, num_classes)

        Returns
        -------
        dict with keys: d_loss, d2d_loss, d2d2_loss, g_loss, g_loss2d,
                        g_loss2d2, g_loss_combined
        """
        real_signals = real_signals.to(self.device)
        real_derivs = real_derivs.to(self.device)
        real_derivs2 = real_derivs2.to(self.device)
        class_vector = class_vector.to(self.device)
        batch = real_signals.size(0)

        # --- Discriminator steps ---
        for _ in range(self.d_steps):
            fake, fake_d1, fake_d2 = self._fake_and_derivs(batch, class_vector)
            fake, fake_d1, fake_d2 = fake.detach(), fake_d1.detach(), fake_d2.detach()

            # Signal discriminator
            gp = _gradient_penalty(self.discriminator, real_signals, fake, class_vector, self.device)
            d_loss = wasserstein_discriminator_loss(
                self.discriminator(real_signals, class_vector),
                self.discriminator(fake, class_vector),
            ) + self.gp_weight * gp
            self.d_opt.zero_grad(); d_loss.backward(); self.d_opt.step()

            # First derivative discriminator
            gp2d = _gradient_penalty(self.deriv_discriminator, real_derivs, fake_d1, class_vector, self.device)
            d2d_loss = wasserstein_discriminator_loss(
                self.deriv_discriminator(real_derivs, class_vector),
                self.deriv_discriminator(fake_d1, class_vector),
            ) + self.gp_weight * gp2d
            self.d2d_opt.zero_grad(); d2d_loss.backward(); self.d2d_opt.step()

            # Second derivative discriminator
            gp2d2 = _gradient_penalty(self.deriv2_discriminator, real_derivs2, fake_d2, class_vector, self.device)
            d2d2_loss = wasserstein_discriminator_loss(
                self.deriv2_discriminator(real_derivs2, class_vector),
                self.deriv2_discriminator(fake_d2, class_vector),
            ) + self.gp_weight * gp2d2
            self.d2d2_opt.zero_grad(); d2d2_loss.backward(); self.d2d2_opt.step()

        # --- Generator step ---
        fake, fake_d1, fake_d2 = self._fake_and_derivs(batch, class_vector)

        g_loss = wasserstein_generator_loss(self.discriminator(fake, class_vector))
        g_loss2d = wasserstein_generator_loss(self.deriv_discriminator(fake_d1, class_vector))
        g_loss2d2 = wasserstein_generator_loss(self.deriv2_discriminator(fake_d2, class_vector))
        g_loss_combined = (g_loss + g_loss2d + g_loss2d2) / 3

        self.g_opt.zero_grad()
        g_loss_combined.backward()
        self.g_opt.step()

        return {
            "d_loss": d_loss.item(),
            "d2d_loss": d2d_loss.item(),
            "d2d2_loss": d2d2_loss.item(),
            "g_loss": g_loss.item(),
            "g_loss2d": g_loss2d.item(),
            "g_loss2d2": g_loss2d2.item(),
            "g_loss_combined": g_loss_combined.item(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_gan(variant="cDVGAN", signal_length=8192, num_classes=NUM_CLASSES,
              noise_dim=100, d_steps=5, gp_weight=10.0, lr=1e-4, device="cpu"):
    """Instantiate a GAN variant by name.

    Parameters
    ----------
    variant : str
        One of ``"cWGAN"``, ``"cDVGAN"``, ``"cDVGAN2"``.

    Returns
    -------
    nn.Module
    """
    kwargs = dict(signal_length=signal_length, num_classes=num_classes,
                  noise_dim=noise_dim, d_steps=d_steps, gp_weight=gp_weight,
                  lr=lr, device=device)
    registry = {"cWGAN": cWGAN, "cDVGAN": cDVGAN, "cDVGAN2": cDVGAN2}
    if variant not in registry:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(registry)}")
    return registry[variant](**kwargs)
