"""PyTorch model components for cDVGAN.

Includes:
- Generator
- Discriminator (operates on raw signals)
- DerivativeDiscriminator (operates on first derivative of signals)
- SecondDerivativeDiscriminator (operates on second derivative of signals)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 7


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """1D conv block with optional batchnorm and dropout."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 use_bn=False, use_dropout=False, drop_value=0.5):
        super().__init__()
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=kernel_size // 2, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            layers.append(nn.Dropout(drop_value))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Upsample + Conv1d block with optional batchnorm and dropout.

    Uses asymmetric padding to match TF padding="same" with stride=1:
      pad_left  = (kernel_size - 1) // 2
      pad_right = kernel_size - 1 - pad_left
    For even kernel_size this differs from symmetric padding=k//2, keeping
    intermediate sequence lengths exact powers of two throughout the generator.
    """

    def __init__(self, in_channels, out_channels, kernel_size=18, up_size=2,
                 use_bn=False, use_dropout=False, drop_value=0.3, activation=None):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=up_size)
        # padding=0 here; asymmetric pad applied manually in forward
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1, padding=0, bias=not use_bn)
        self._pad_left  = (kernel_size - 1) // 2
        self._pad_right = kernel_size - 1 - self._pad_left
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.activation = activation if activation is not None else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_value) if use_dropout else None

    def forward(self, x):
        x = self.upsample(x)
        x = F.pad(x, (self._pad_left, self._pad_right))
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Conditional generator.

    Takes a noise vector and a class vector (one-hot or soft), outputs a
    time series of length ``signal_length``.

    Parameters
    ----------
    noise_dim : int
        Dimension of the input noise vector.
    num_classes : int
        Number of glitch classes.
    signal_length : int
        Length of the output time series.
    """

    def __init__(self, noise_dim=100, num_classes=NUM_CLASSES, signal_length=8192):
        super().__init__()
        self.signal_length = signal_length

        # Project class vector to 32-dim embedding
        self.class_embedding = nn.Linear(num_classes, 32, bias=False)

        # Initial dense projection: (noise_dim + 32) -> 256 * 16
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + 32, 256 * 16, bias=False),
            nn.ReLU(inplace=True),
        )

        # Upsample blocks: (batch, 16, 256) -> (batch, 1, signal_length)
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(16, 512, use_bn=True),
            UpsampleBlock(512, 256, use_bn=True),
            UpsampleBlock(256, 128, use_bn=True),
            UpsampleBlock(128, 64, use_bn=True),
            UpsampleBlock(64, 1, use_bn=False, activation=nn.Identity()),
        )

        self.output_length = signal_length

    def forward(self, noise, class_vector):
        """
        Parameters
        ----------
        noise : Tensor (batch, noise_dim)
        class_vector : Tensor (batch, num_classes)

        Returns
        -------
        Tensor (batch, signal_length)
        """
        class_emb = self.class_embedding(class_vector)          # (batch, 32)
        x = torch.cat([noise, class_emb], dim=1)                # (batch, noise_dim+32)
        x = self.fc(x)                                          # (batch, 256*16)
        x = x.view(x.size(0), 16, 256)                         # (batch, 16, 256)
        x = self.upsample_blocks(x)                             # (batch, 1, signal_length)
        x = x[:, 0, :]                                          # (batch, signal_length)
        return x


# ---------------------------------------------------------------------------
# Discriminator (operates on raw signals)
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """Conditional discriminator using projection conditioning.

    Parameters
    ----------
    signal_length : int
        Length of the input time series.
    num_classes : int
        Number of glitch classes.
    """

    def __init__(self, signal_length=8192, num_classes=NUM_CLASSES):
        super().__init__()

        # Reshape input: (batch, signal_length) -> (batch, 1, signal_length)
        # then treat as (batch, 128, 64) after initial reshape
        self.conv_blocks = nn.Sequential(
            ConvBlock(128, 128, kernel_size=14, stride=1, use_dropout=True),
            ConvBlock(128, 128, kernel_size=14, stride=2, use_dropout=True),
            ConvBlock(128, 256, kernel_size=14, stride=2, use_dropout=True),
            ConvBlock(256, 256, kernel_size=14, stride=2, use_dropout=True),
            ConvBlock(256, 512, kernel_size=14, stride=2, use_dropout=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 128, bias=False)

        # Projection conditioning
        self.class_embedding = nn.Embedding(num_classes, 128)
        self.scalar = nn.Linear(128, 1, bias=False)

    def forward(self, signal, class_vector):
        """
        Parameters
        ----------
        signal : Tensor (batch, signal_length)
        class_vector : Tensor (batch, num_classes)  one-hot or soft

        Returns
        -------
        Tensor (batch, 1)
        """
        batch = signal.size(0)
        x = signal.view(batch, 128, -1)         # (batch, 128, 64)
        x = self.conv_blocks(x)                 # (batch, 512, T')
        x = self.pool(x).squeeze(-1)            # (batch, 512)
        x = self.fc(x)                          # (batch, 128)

        # Projection conditioning: dot(x, embed(class))
        class_idx = class_vector.argmax(dim=-1)             # (batch,)
        class_emb = self.class_embedding(class_idx)         # (batch, 128)
        dot = (x * class_emb).sum(dim=1, keepdim=True)     # (batch, 1)
        scalar = self.scalar(x)                             # (batch, 1)
        return dot + scalar


# ---------------------------------------------------------------------------
# Derivative discriminator (operates on first derivative of signals)
# ---------------------------------------------------------------------------

class DerivativeDiscriminator(nn.Module):
    """Discriminator that operates on the first derivative of signals.

    Parameters
    ----------
    signal_length : int
        Length of the *derivative* input (signal_length - 1).
    num_classes : int
        Number of glitch classes.
    """

    def __init__(self, signal_length=8191, num_classes=NUM_CLASSES):
        super().__init__()

        self.fc_in = nn.Sequential(
            nn.Linear(signal_length, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            ConvBlock(8, 64, kernel_size=5, stride=1, use_dropout=True),
            ConvBlock(64, 128, kernel_size=5, stride=2, use_dropout=True),
            ConvBlock(128, 256, kernel_size=5, stride=2, use_dropout=True),
            ConvBlock(256, 256, kernel_size=5, stride=2, use_dropout=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 128, bias=False)

        self.class_embedding = nn.Embedding(num_classes, 128)
        self.scalar = nn.Linear(128, 1, bias=False)

    def forward(self, signal_deriv, class_vector):
        """
        Parameters
        ----------
        signal_deriv : Tensor (batch, signal_length-1)
        class_vector : Tensor (batch, num_classes)

        Returns
        -------
        Tensor (batch, 1)
        """
        x = self.fc_in(signal_deriv)            # (batch, 256)
        x = x.view(x.size(0), 8, 32)           # (batch, 8, 32)
        x = self.conv_blocks(x)                 # (batch, 256, T')
        x = self.pool(x).squeeze(-1)            # (batch, 256)
        x = self.fc(x)                          # (batch, 128)

        class_idx = class_vector.argmax(dim=-1)
        class_emb = self.class_embedding(class_idx)
        dot = (x * class_emb).sum(dim=1, keepdim=True)
        scalar = self.scalar(x)
        return dot + scalar


# ---------------------------------------------------------------------------
# Second derivative discriminator
# ---------------------------------------------------------------------------

class SecondDerivativeDiscriminator(nn.Module):
    """Discriminator that operates on the second derivative of signals.

    Parameters
    ----------
    signal_length : int
        Length of the *second derivative* input (signal_length - 2).
    num_classes : int
        Number of glitch classes.
    """

    def __init__(self, signal_length=8190, num_classes=NUM_CLASSES):
        super().__init__()

        self.fc_in = nn.Sequential(
            nn.Linear(signal_length, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            ConvBlock(32, 64, kernel_size=5, stride=1, use_dropout=True),
            ConvBlock(64, 128, kernel_size=5, stride=2, use_dropout=True),
            ConvBlock(128, 256, kernel_size=5, stride=2, use_dropout=True),
            ConvBlock(256, 256, kernel_size=5, stride=2, use_dropout=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 128, bias=False)

        self.class_embedding = nn.Embedding(num_classes, 128)
        self.scalar = nn.Linear(128, 1, bias=False)

    def forward(self, signal_deriv2, class_vector):
        """
        Parameters
        ----------
        signal_deriv2 : Tensor (batch, signal_length-2)
        class_vector : Tensor (batch, num_classes)

        Returns
        -------
        Tensor (batch, 1)
        """
        x = self.fc_in(signal_deriv2)           # (batch, 512)
        x = x.view(x.size(0), 32, 16)          # (batch, 32, 16)
        x = self.conv_blocks(x)                 # (batch, 256, T')
        x = self.pool(x).squeeze(-1)            # (batch, 256)
        x = self.fc(x)                          # (batch, 128)

        class_idx = class_vector.argmax(dim=-1)
        class_emb = self.class_embedding(class_idx)
        dot = (x * class_emb).sum(dim=1, keepdim=True)
        scalar = self.scalar(x)
        return dot + scalar
