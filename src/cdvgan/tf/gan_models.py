"""TensorFlow/Keras GAN model classes for cDVGAN.

Includes:
- cWGAN   — conditional Wasserstein GAN with gradient penalty
- cDVGAN  — adds a first-derivative discriminator
- cDVGAN2 — adds first and second derivative discriminators
- build_gan() — factory function
"""

import tensorflow as tf
import keras
from keras.optimizers import RMSprop

from cdvgan.tf.model_components import (
    get_discriminator_model,
    get_derivative_discriminator_model,
    get_second_derivative_discriminator_model,
    get_generator_model,
)

NUM_CLASSES = 7


def _diff(x):
    """First-order finite difference along last axis. Replaces calculate_derivative."""
    return x[..., 1:] - x[..., :-1]


def _gradient_penalty(discriminator, real, fake, classes, batch_size):
    """WGAN-GP gradient penalty for a given discriminator."""
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0, dtype=tf.float32)
    interpolated = real + alpha * (fake - real)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator([interpolated, classes], training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads)))
    return tf.reduce_mean((norm - 1.0) ** 2)


# ---------------------------------------------------------------------------
# cWGAN
# ---------------------------------------------------------------------------

class cWGAN(keras.Model):
    """Conditional Wasserstein GAN with gradient penalty."""

    def __init__(self, signal_length=8192, num_classes=NUM_CLASSES, noise_dim=100,
                 d_steps=5, gp_weight=10.0, lr=1e-4):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.d_steps = d_steps
        self.gp_weight = gp_weight

        self.generator = get_generator_model(noise_dim, num_classes)
        self.discriminator = get_discriminator_model(signal_length, num_classes)

        self.g_optimizer = RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7)
        self.d_optimizer = RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7)

    def train_step(self, data):
        # data is (signals_batch, classes_batch) from tf.data.Dataset
        real_signals = tf.cast(data[0], tf.float32)
        class_array  = tf.cast(data[1], tf.float32)
        batch_size = tf.shape(real_signals)[0]

        for _ in range(self.d_steps):
            noise = tf.random.normal((batch_size, self.noise_dim))
            with tf.GradientTape() as tape:
                fake = self.generator([noise, class_array], training=True)
                real_logits = self.discriminator([real_signals, class_array], training=True)
                fake_logits = self.discriminator([fake, class_array], training=True)
                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = _gradient_penalty(self.discriminator, real_signals, fake,
                                       class_array, batch_size)
                d_loss = d_cost + self.gp_weight * gp
            grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))

        noise = tf.random.normal((batch_size, self.noise_dim))
        with tf.GradientTape() as tape:
            fake = self.generator([noise, class_array], training=True)
            fake_logits = self.discriminator([fake, class_array], training=True)
            g_loss = -tf.reduce_mean(fake_logits)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}


# ---------------------------------------------------------------------------
# cDVGAN
# ---------------------------------------------------------------------------

class cDVGAN(keras.Model):
    """Conditional Dual-discriminator Variational GAN (first derivative)."""

    def __init__(self, signal_length=8192, num_classes=NUM_CLASSES, noise_dim=100,
                 d_steps=5, gp_weight=10.0, lr=1e-4):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.d_steps = d_steps
        self.gp_weight = gp_weight

        self.generator = get_generator_model(noise_dim, num_classes)
        self.discriminator = get_discriminator_model(signal_length, num_classes)
        self.deriv_discriminator = get_derivative_discriminator_model(
            signal_length - 1, num_classes)

        self.g_optimizer   = RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7)
        self.d_optimizer   = RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7)
        self.d2d_optimizer = RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7)

    def train_step(self, data):
        # data is (signals_batch, derivs_batch, classes_batch)
        real_signals = tf.cast(data[0], tf.float32)
        real_derivs  = tf.cast(data[1], tf.float32)
        class_array  = tf.cast(data[2], tf.float32)
        batch_size = tf.shape(real_signals)[0]

        for _ in range(self.d_steps):
            noise = tf.random.normal((batch_size, self.noise_dim))
            with tf.GradientTape(persistent=True) as tape:
                fake = self.generator([noise, class_array], training=True)
                fake_derivs = _diff(fake)

                real_logits    = self.discriminator([real_signals, class_array], training=True)
                fake_logits    = self.discriminator([fake, class_array], training=True)
                real_logits2d  = self.deriv_discriminator([real_derivs, class_array], training=True)
                fake_logits2d  = self.deriv_discriminator([fake_derivs, class_array], training=True)

                d_cost   = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                d2d_cost = tf.reduce_mean(fake_logits2d) - tf.reduce_mean(real_logits2d)

                gp   = _gradient_penalty(self.discriminator, real_signals, fake,
                                         class_array, batch_size)
                gp2d = _gradient_penalty(self.deriv_discriminator, real_derivs, fake_derivs,
                                         class_array, batch_size)

                d_loss   = d_cost   + self.gp_weight * gp
                d2d_loss = d2d_cost + self.gp_weight * gp2d

            d_grads   = tape.gradient(d_loss,   self.discriminator.trainable_variables)
            d2d_grads = tape.gradient(d2d_loss, self.deriv_discriminator.trainable_variables)
            del tape

            self.d_optimizer.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables))
            self.d2d_optimizer.apply_gradients(
                zip(d2d_grads, self.deriv_discriminator.trainable_variables))

        noise = tf.random.normal((batch_size, self.noise_dim))
        with tf.GradientTape() as tape:
            fake = self.generator([noise, class_array], training=True)
            fake_derivs = _diff(fake)
            g_loss    = -tf.reduce_mean(self.discriminator([fake, class_array], training=True))
            g_loss2d  = -tf.reduce_mean(self.deriv_discriminator([fake_derivs, class_array], training=True))
            g_loss_combined = 0.5 * (g_loss + g_loss2d)
        grads = tape.gradient(g_loss_combined, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {
            "d_loss": d_loss, "d2d_loss": d2d_loss,
            "g_loss": g_loss, "g_loss2d": g_loss2d,
            "g_loss_combined": g_loss_combined,
        }


# ---------------------------------------------------------------------------
# cDVGAN2
# ---------------------------------------------------------------------------

class cDVGAN2(keras.Model):
    """cDVGAN with an additional second-derivative discriminator."""

    def __init__(self, signal_length=8192, num_classes=NUM_CLASSES, noise_dim=100,
                 d_steps=5, gp_weight=10.0, lr=1e-4):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.d_steps = d_steps
        self.gp_weight = gp_weight

        self.generator = get_generator_model(noise_dim, num_classes)
        self.discriminator = get_discriminator_model(signal_length, num_classes)
        self.deriv_discriminator  = get_derivative_discriminator_model(
            signal_length - 1, num_classes)
        self.deriv2_discriminator = get_second_derivative_discriminator_model(
            signal_length - 2, num_classes)

        self.g_optimizer    = RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7)
        self.d_optimizer    = RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7)
        self.d2d_optimizer  = RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7)
        self.d2d2_optimizer = RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-7)

    def train_step(self, data):
        # data is (signals_batch, derivs_batch, derivs2_batch, classes_batch)
        real_signals = tf.cast(data[0], tf.float32)
        real_derivs  = tf.cast(data[1], tf.float32)
        real_derivs2 = tf.cast(data[2], tf.float32)
        class_array  = tf.cast(data[3], tf.float32)
        batch_size = tf.shape(real_signals)[0]

        for _ in range(self.d_steps):
            noise = tf.random.normal((batch_size, self.noise_dim))
            with tf.GradientTape(persistent=True) as tape:
                fake         = self.generator([noise, class_array], training=True)
                fake_derivs  = _diff(fake)
                fake_derivs2 = _diff(fake_derivs)

                real_logits    = self.discriminator([real_signals, class_array], training=True)
                fake_logits    = self.discriminator([fake, class_array], training=True)
                real_logits2d  = self.deriv_discriminator([real_derivs, class_array], training=True)
                fake_logits2d  = self.deriv_discriminator([fake_derivs, class_array], training=True)
                real_logits2d2 = self.deriv2_discriminator([real_derivs2, class_array], training=True)
                fake_logits2d2 = self.deriv2_discriminator([fake_derivs2, class_array], training=True)

                d_cost    = tf.reduce_mean(fake_logits)    - tf.reduce_mean(real_logits)
                d2d_cost  = tf.reduce_mean(fake_logits2d)  - tf.reduce_mean(real_logits2d)
                d2d2_cost = tf.reduce_mean(fake_logits2d2) - tf.reduce_mean(real_logits2d2)

                gp    = _gradient_penalty(self.discriminator, real_signals, fake,
                                          class_array, batch_size)
                gp2d  = _gradient_penalty(self.deriv_discriminator, real_derivs, fake_derivs,
                                          class_array, batch_size)
                gp2d2 = _gradient_penalty(self.deriv2_discriminator, real_derivs2, fake_derivs2,
                                          class_array, batch_size)

                d_loss    = d_cost    + self.gp_weight * gp
                d2d_loss  = d2d_cost  + self.gp_weight * gp2d
                d2d2_loss = d2d2_cost + self.gp_weight * gp2d2

            d_grads    = tape.gradient(d_loss,    self.discriminator.trainable_variables)
            d2d_grads  = tape.gradient(d2d_loss,  self.deriv_discriminator.trainable_variables)
            d2d2_grads = tape.gradient(d2d2_loss, self.deriv2_discriminator.trainable_variables)
            del tape

            self.d_optimizer.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables))
            self.d2d_optimizer.apply_gradients(
                zip(d2d_grads, self.deriv_discriminator.trainable_variables))
            self.d2d2_optimizer.apply_gradients(
                zip(d2d2_grads, self.deriv2_discriminator.trainable_variables))

        noise = tf.random.normal((batch_size, self.noise_dim))
        with tf.GradientTape() as tape:
            fake         = self.generator([noise, class_array], training=True)
            fake_derivs  = _diff(fake)
            fake_derivs2 = _diff(fake_derivs)
            g_loss    = -tf.reduce_mean(self.discriminator([fake, class_array], training=True))
            g_loss2d  = -tf.reduce_mean(self.deriv_discriminator([fake_derivs, class_array], training=True))
            g_loss2d2 = -tf.reduce_mean(self.deriv2_discriminator([fake_derivs2, class_array], training=True))
            g_loss_combined = (g_loss + g_loss2d + g_loss2d2) / 3.0
        grads = tape.gradient(g_loss_combined, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {
            "d_loss": d_loss, "d2d_loss": d2d_loss, "d2d2_loss": d2d2_loss,
            "g_loss": g_loss, "g_loss2d": g_loss2d, "g_loss2d2": g_loss2d2,
            "g_loss_combined": g_loss_combined,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_gan(variant="cDVGAN", signal_length=8192, num_classes=NUM_CLASSES,
              noise_dim=100, d_steps=5, gp_weight=10.0, lr=1e-4):
    """Instantiate a TF GAN variant by name.

    Parameters
    ----------
    variant : str
        One of ``"cWGAN"``, ``"cDVGAN"``, ``"cDVGAN2"``.

    Returns
    -------
    keras.Model
    """
    kwargs = dict(signal_length=signal_length, num_classes=num_classes,
                  noise_dim=noise_dim, d_steps=d_steps, gp_weight=gp_weight, lr=lr)
    registry = {"cWGAN": cWGAN, "cDVGAN": cDVGAN, "cDVGAN2": cDVGAN2}
    if variant not in registry:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(registry)}")
    return registry[variant](**kwargs)
