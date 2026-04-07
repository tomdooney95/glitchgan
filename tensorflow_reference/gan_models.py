import numpy as np
import tensorflow as tf
import keras
from utils import calculate_derivative
from tensorflow.keras.optimizers import RMSprop, Adam
from model_components import *
from utils import *
num_classes = 7

class cWGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0,
    ):
        super(cWGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(cWGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_signals, classes, fake_signals):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated signal
        and added to the discriminator loss.
        """
        # Get the interpolated signal
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0, dtype=tf.float32)
        diff = fake_signals - real_signals
        interpolated = real_signals + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated signal.
            pred = self.discriminator([interpolated, classes], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=None))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_step(self, data):
        real_signals = tf.cast(data[0][0], dtype = tf.float32)
        class_array = tf.cast(data[0][1], dtype = tf.float32)
        # Get the batch size
        batch_size = tf.shape(real_signals)[0]
        
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake signals from the latent vector
                fake_signals = self.generator([random_latent_vectors, class_array], training=True)
                # Get the logits for the fake signals
                fake_logits = self.discriminator([fake_signals, class_array], training=True)
                # Get the logits for the real signals
                real_logits = self.discriminator([real_signals, class_array], training=True)

                # Calculate the discriminator loss using the fake and real signal logits
                d_cost = self.d_loss_fn(real_sig=real_logits, fake_sig=fake_logits)
                # Calculate the gradient penalty
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_signals, class_array, fake_signals)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake signals using the generator
            generated_signals = self.generator([random_latent_vectors, class_array], training=True)
            # Get the discriminator logits for fake signals
            gen_sig_logits = self.discriminator([generated_signals, class_array], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_sig_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

class cDVGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        deriv_discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0,
    ):
        super(cDVGAN, self).__init__()
        self.discriminator = discriminator
        self.deriv_discriminator = deriv_discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, d2d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(cDVGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.d2d_optimizer = d2d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.d2d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_signals, classes, fake_signals):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated signal
        and added to the discriminator loss.
        """
        # Get the interpolated signal
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0, dtype=tf.float32)
        diff = fake_signals - real_signals
        interpolated = real_signals + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated signal.
            pred = self.discriminator([interpolated, classes], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=None))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    def gradient_penalty_derivatives(self, batch_size, real_signals, classes, fake_signals):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated signal
        and added to the discriminator loss.
        """
        # Get the interpolated signal
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0, dtype=tf.float32)
        diff = fake_signals - real_signals
        interpolated = real_signals + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated signal.
            pred = self.deriv_discriminator([interpolated, classes], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=None))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_signals = tf.cast(data[0][0], dtype = tf.float32)
        real_derivatives= tf.cast(data[0][1], dtype = tf.float32)
        class_array = tf.cast(data[0][2], dtype = tf.float32)
        
        # Get the batch size
        batch_size = tf.shape(real_signals)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminators and get the discriminator losses
        # 3. Calculate the gradient penalties
        # 4. Multiply this gradient penalties with a constant weight factor
        # 5. Add the respective gradient penalties to each discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape(persistent=True) as tape:
                # Generate fake signals from the latent vector
                fake_signals = self.generator([random_latent_vectors, class_array], training=True)
                # fake_signals = tf.cast(fake_signals, dtype=tf.float32)
                
                fake_derivatives = calculate_derivative(tf.cast(np.array(range(8192)), tf.float32), fake_signals)
                # fake_derivatives = tf.cast(fake_derivatives, dtype=tf.float32)
     
                # Get the logits for the fake signals
                fake_logits = self.discriminator([fake_signals, class_array], training=True)
                # Get the logits for the real signals
                real_logits = self.discriminator([real_signals, class_array], training=True)

                # Get the logits for the fake derivatives
                fake_logits2d = self.deriv_discriminator([fake_derivatives, class_array], training=True)
                # Get the logits for the real derivatives
                real_logits2d = self.deriv_discriminator([real_derivatives, class_array], training=True)

                # Calculate the discriminator loss using the fake and real signal logits
                d_cost = self.d_loss_fn(real_sig=real_logits, fake_sig=fake_logits)

                # Calculate the derivative discriminator loss using the fake and real signal logits
                d2d_cost = self.d_loss_fn(real_sig=real_logits2d, fake_sig=fake_logits2d)

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_signals, class_array, fake_signals)

                # Calculate the gradient penalty
                gp2d = self.gradient_penalty_derivatives(batch_size, real_derivatives, class_array, fake_derivatives)

                # Add the gradient penalty to the original discriminator loss
                # d_loss = tf.cast(d_cost, dtype=tf.float32) + gp * self.gp_weight
                # d2d_loss = tf.cast(d2d_cost, dtype=tf.float32) + gp2d * self.gp_weight
                
                d_loss = d_cost + gp * self.gp_weight
                d2d_loss = d2d_cost + gp2d * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)

            # Get the gradients w.r.t the deriv_discriminator loss
            d2d_gradient = tape.gradient(d2d_loss, self.deriv_discriminator.trainable_variables)

            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

            # Update the weights of the deriv_discriminator using the deriv_discriminator optimizer
            self.d2d_optimizer.apply_gradients(
                zip(d2d_gradient, self.deriv_discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake signals using the generator
            generated_signals = self.generator([random_latent_vectors, class_array], training=True)

            # Calculate derivatives from fake signals
            generated_derivatives = calculate_derivative(tf.cast(np.array(range(8192)), tf.float32), generated_signals)

            # Get the discriminator logits for fake signals
            gen_sig_logits = self.discriminator([generated_signals, class_array], training=True)
            # Get the deriv_discriminator logits for fake derivatives
            gen_sig2d_logits = self.deriv_discriminator([generated_derivatives, class_array], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_sig_logits)

            g_loss2d = self.g_loss_fn(gen_sig2d_logits)

            g_loss_combined = (1/2)*(g_loss+g_loss2d)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss_combined, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "d2d_loss": d2d_loss, "g_loss": g_loss, "g_loss2d":g_loss2d, "g_loss_combined":g_loss_combined}

class cDVGAN2(keras.Model):
    def __init__(
        self,
        discriminator,
        deriv_discriminator,
        deriv2_discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0,
    ):
        super(cDVGAN2, self).__init__()
        self.discriminator = discriminator
        self.deriv_discriminator = deriv_discriminator
        self.deriv2_discriminator = deriv2_discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, d2d_optimizer, d2d2_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(cDVGAN2, self).compile()
        self.d_optimizer = d_optimizer
        self.d2d_optimizer = d2d_optimizer
        self.d2d2_optimizer = d2d2_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.d2d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_signals, classes, fake_signals):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated signal
        and added to the discriminator loss.
        """
        # Get the interpolated signal
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0, dtype=tf.float32)
        diff = fake_signals - real_signals
        interpolated = real_signals + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated signal.
            pred = self.discriminator([interpolated, classes], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=None))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    def gradient_penalty_derivatives(self, batch_size, real_signals, classes, fake_signals):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated signal
        and added to the discriminator loss.
        """
        # Get the interpolated signal
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0, dtype=tf.float32)
        diff = fake_signals - real_signals
        interpolated = real_signals + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated signal.
            pred = self.deriv_discriminator([interpolated, classes], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=None))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def gradient_penalty_derivatives2(self, batch_size, real_signals, classes, fake_signals):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated signal
        and added to the discriminator loss.
        """
        # Get the interpolated signal
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0, dtype=tf.float32)
        diff = fake_signals - real_signals
        interpolated = real_signals + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated signal.
            pred = self.deriv2_discriminator([interpolated, classes], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated signal.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=None))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_signals = tf.cast(data[0][0], dtype = tf.float32)
        real_derivatives= tf.cast(data[0][1], dtype = tf.float32)
        real_derivatives2= tf.cast(data[0][2], dtype = tf.float32)
        class_array = tf.cast(data[0][3], dtype = tf.float32)
        
        # Get the batch size
        batch_size = tf.shape(real_signals)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminators and get the discriminator losses
        # 3. Calculate the gradient penalties
        # 4. Multiply this gradient penalties with a constant weight factor
        # 5. Add the respective gradient penalties to each discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape(persistent=True) as tape:
                # Generate fake signals from the latent vector
                fake_signals = self.generator([random_latent_vectors, class_array], training=True)
                # fake_signals = tf.cast(fake_signals, dtype=tf.float32)
                
                fake_derivatives = calculate_derivative(tf.cast(np.array(range(8192)), tf.float32), fake_signals)
                fake_derivatives2 = calculate_derivative(tf.cast(np.array(range(1023)), tf.float32), fake_derivatives)
     
                # Get the logits for the fake signals
                fake_logits = self.discriminator([fake_signals, class_array], training=True)
                # Get the logits for the real signals
                real_logits = self.discriminator([real_signals, class_array], training=True)

                # Get the logits for the fake derivatives
                fake_logits2d = self.deriv_discriminator([fake_derivatives, class_array], training=True)
                # Get the logits for the real derivatives
                real_logits2d = self.deriv_discriminator([real_derivatives, class_array], training=True)
                
                # Get the logits for the fake 2nd order derivatives
                fake_logits2d2 = self.deriv2_discriminator([fake_derivatives2, class_array], training=True)
                # Get the logits for the real 2nd order derivatives
                real_logits2d2 = self.deriv2_discriminator([real_derivatives2, class_array], training=True)

                # Calculate the discriminator loss using the fake and real signal logits
                d_cost = self.d_loss_fn(real_sig=real_logits, fake_sig=fake_logits)

                # Calculate the derivative discriminator loss using the fake and real signal logits
                d2d_cost = self.d_loss_fn(real_sig=real_logits2d, fake_sig=fake_logits2d)
                
                # Calculate the derivative discriminator loss using the fake and real signal logits
                d2d2_cost = self.d_loss_fn(real_sig=real_logits2d2, fake_sig=fake_logits2d2)

                # Calculate the gradient penalty for 1st order
                gp = self.gradient_penalty(batch_size, real_signals, class_array, fake_signals)

                # Calculate the gradient penalty
                gp2d = self.gradient_penalty_derivatives(batch_size, real_derivatives, class_array, fake_derivatives)
                
                # And again, for 2nd order
                gp2d2 = self.gradient_penalty_derivatives2(batch_size, real_derivatives2, class_array, fake_derivatives2)

                # Add the gradient penalty to the original discriminator loss
                # d_loss = tf.cast(d_cost, dtype=tf.float32) + gp * self.gp_weight
                # d2d_loss = tf.cast(d2d_cost, dtype=tf.float32) + gp2d * self.gp_weight
                
                d_loss = d_cost + gp * self.gp_weight
                d2d_loss = d2d_cost + gp2d * self.gp_weight
                d2d2_loss = d2d2_cost + gp2d2 * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)

            # Get the gradients w.r.t the deriv_discriminator loss
            d2d_gradient = tape.gradient(d2d_loss, self.deriv_discriminator.trainable_variables)
            
            # Get the gradients w.r.t the 2nd order deriv_discriminator loss
            d2d2_gradient = tape.gradient(d2d2_loss, self.deriv2_discriminator.trainable_variables)

            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

            # Update the weights of the deriv_discriminator using the deriv_discriminator optimizer
            self.d2d_optimizer.apply_gradients(
                zip(d2d_gradient, self.deriv_discriminator.trainable_variables)
            )
            
            # Update the weights of the 2nd order deriv_discriminator using the deriv2_discriminator optimizer
            self.d2d2_optimizer.apply_gradients(
                zip(d2d2_gradient, self.deriv2_discriminator.trainable_variables)
            )


        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake signals using the generator
            generated_signals = self.generator([random_latent_vectors, class_array], training=True)

            # generated_signals = tf.cast(generated_signals, dtype=tf.float32)

            # Calculate derivatives from fake signals

            generated_derivatives = calculate_derivative(tf.cast(np.array(range(8192)), tf.float32), generated_signals)
            generated_derivatives2 = calculate_derivative(tf.cast(np.array(range(8191)), tf.float32), generated_derivatives)

            # Get the discriminator logits for fake signals
            gen_sig_logits = self.discriminator([generated_signals, class_array], training=True)
            # Get the deriv_discriminator logits for fake derivatives
            gen_sig2d_logits = self.deriv_discriminator([generated_derivatives, class_array], training=True)
            
            # Get the deriv2_discriminator logits for fake 2nd order derivatives
            gen_sig2d2_logits = self.deriv2_discriminator([generated_derivatives2, class_array], training=True)
            
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_sig_logits)

            g_loss2d = self.g_loss_fn(gen_sig2d_logits)
            
            g_loss2d2 = self.g_loss_fn(gen_sig2d2_logits)

            g_loss_combined = (1/3)*(g_loss+g_loss2d+g_loss2d2)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss_combined, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "d2d_loss": d2d_loss, "d2d2_loss": d2d2_loss,  "g_loss": g_loss, "g_loss2d":g_loss2d, "g_loss2d2":g_loss2d2, "g_loss_combined":g_loss_combined}

class MCGANN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=1,
        gp_weight=10.0,
    ):
        super(MCGANN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(MCGANN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.d2d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        real_signals = tf.cast(data[0][0], dtype = tf.float32)
        class_array = tf.cast(data[0][1], dtype = tf.float32)

        # Get the batch size
        batch_size = tf.shape(real_signals)[0]
        batch_size_np = np.shape(real_signals)[0]
        
        real_signals = tf.cast(real_signals, dtype=tf.float32)

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
          # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape(persistent=True) as tape:
                # Generate fake signals from the latent vector
                fake_signals = self.generator([random_latent_vectors, class_array], training=True)

                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_signals, class_array], training=True)
                # Get the logits for the real images
                real_logits = self.discriminator([real_signals, class_array], training=True)
                
                y_real = tf.ones((batch_size, 1))
                y_fake = tf.zeros((batch_size,1))

                # Calculate the discriminator loss using the fake and real image logits
                d_loss_real = self.d_loss_fn(y_real, real_logits)

                d_loss_fake = self.d_loss_fn(y_fake, fake_logits)

                d_loss = (d_loss_real + d_loss_fake)/2

            # Get the gradients w.r.t the discriminator loss
            d_gradient_real = tape.gradient(d_loss_real, self.discriminator.trainable_variables)

            d_gradient_fake = tape.gradient(d_loss_fake, self.discriminator.trainable_variables)

            # print('Model Types:', type(self.discriminator), type(self.discriminator2d))

            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient_real, self.discriminator.trainable_variables)
            )
            self.d_optimizer.apply_gradients(
                zip(d_gradient_fake, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(2*batch_size, self.latent_dim))
        indices = tf.experimental.numpy.random.randint(
                0,
                high=num_classes,
                size=[2*batch_size])
        depth = num_classes
        random_classes = tf.one_hot(indices, depth,
                  on_value=1.0, off_value=0.0,
                  axis=-1)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_signals = self.generator([random_latent_vectors, random_classes], training=True)
            # Get the discriminator logits for fake images
            gen_sig_logits = self.discriminator([generated_signals, random_classes], training=True)
            # Calculate the generator loss

            # create inverted labels for the fakes, this is how G tries to trick D
            y_gan = tf.ones((2*batch_size,1))

            g_loss = self.g_loss_fn(y_gan, gen_sig_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

        
class MCDVGANN(keras.Model):
    def __init__(
        self,
        discriminator,
        deriv_discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=1,
        gp_weight=10.0,
    ):
        super(MCDVGANN, self).__init__()
        self.discriminator = discriminator
        self.deriv_discriminator = deriv_discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, d2d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(MCDVGANN, self).compile()
        self.d_optimizer = d_optimizer
        self.d2d_optimizer = d2d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.d2d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        real_signals = tf.cast(data[0][0], dtype = tf.float32)
        real_derivatives= tf.cast(data[0][1], dtype = tf.float32)
        class_array = tf.cast(data[0][2], dtype = tf.float32)

        # Get the batch size
        batch_size = tf.shape(real_signals)[0]

        real_signals = tf.cast(real_signals, dtype=tf.float32)
        real_derivatives = tf.cast(real_derivatives, dtype=tf.float32)

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
          # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape(persistent=True) as tape:
                # Generate fake signals from the latent vector
                fake_signals = self.generator([random_latent_vectors, class_array], training=True)

                fake_derivatives = calculate_derivative(tf.cast(np.array(range(8192)), tf.float32), fake_signals)

                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_signals, class_array], training=True)
                # Get the logits for the real images
                real_logits = self.discriminator([real_signals, class_array], training=True)

                # Get the logits for the fake spectrograms
                fake_logits2d = self.deriv_discriminator([fake_derivatives, class_array], training=True)
                # Get the logits for the real spectrograms
                real_logits2d = self.deriv_discriminator([real_derivatives, class_array], training=True)

                y_real = tf.ones((batch_size, 1))
                y_fake = tf.zeros((batch_size,1))

                # Calculate the discriminator loss using the fake and real image logits
                d_loss_real = self.d_loss_fn(y_real, real_logits)

                d_loss_fake = self.d_loss_fn(y_fake, fake_logits)

                d_loss = (d_loss_real + d_loss_fake)/2

                # Calculate the discriminator loss using the fake and real image logits
                d2d_loss_real = self.d_loss_fn(y_real, real_logits2d)

                d2d_loss_fake = self.d_loss_fn(y_fake, fake_logits2d)

                d2d_loss = (d2d_loss_real + d2d_loss_fake)/2

            # Get the gradients w.r.t the discriminator loss
            d_gradient_real = tape.gradient(d_loss_real, self.discriminator.trainable_variables)

            d_gradient_fake = tape.gradient(d_loss_fake, self.discriminator.trainable_variables)

            # Get the gradients w.r.t the deriv_discriminator loss
            d2d_gradient_real = tape.gradient(d2d_loss_real, self.deriv_discriminator.trainable_variables)

            d2d_gradient_fake = tape.gradient(d2d_loss_fake, self.deriv_discriminator.trainable_variables)

            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient_real, self.discriminator.trainable_variables)
            )
            self.d_optimizer.apply_gradients(
                zip(d_gradient_fake, self.discriminator.trainable_variables)
            )

            # Update the weights of the deriv_discriminator using the deriv_discriminator optimizer
            self.d2d_optimizer.apply_gradients(
                zip(d2d_gradient_real, self.deriv_discriminator.trainable_variables)
            )
            self.d2d_optimizer.apply_gradients(
                zip(d2d_gradient_fake, self.deriv_discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(2*batch_size, self.latent_dim))
        indices = tf.experimental.numpy.random.randint(
                0,
                high=num_classes,
                size=[2*batch_size])
        depth = num_classes
        random_classes = tf.one_hot(indices, depth,
                  on_value=1.0, off_value=0.0,
                  axis=-1)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_signals = self.generator([random_latent_vectors, random_classes], training=True)
            # Generate derivatives
            generated_derivatives = calculate_derivative(tf.cast(np.array(range(8192)), tf.float32), generated_signals)
            # Get the discriminator logits for fake images
            gen_sig_logits = self.discriminator([generated_signals, random_classes], training=True)

            # Get the deriv_discriminator logits for fake spectograms
            gen_sig2d_logits = self.deriv_discriminator([generated_derivatives, random_classes], training=True)
            # Calculate the generator loss

            # create inverted labels for the fakes, this is how G tries to trick D
            y_gan = tf.ones((2*batch_size,1))

            g_loss = self.g_loss_fn(y_gan, gen_sig_logits)

            g_loss2d = self.g_loss_fn(y_gan, gen_sig2d_logits)

            g_loss_combined = (1/2)*(g_loss+g_loss2d)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss_combined, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "d2d_loss": d2d_loss, "g_loss": g_loss, "g_loss2d":g_loss2d, "g_loss_combined":g_loss_combined}

def choose_gan(gan_choice, signal_length, deriv_signal_length, deriv2_signal_length, num_classes, noise_dim):
    # Instantiating the optimizers.
    generator_optimizer = RMSprop(.0001)
    discriminator_optimizer = RMSprop(.0001)
    discriminator2d_optimizer = RMSprop(.0001)
    discriminator2d2_optimizer = RMSprop(.0001)
    generator_optimizer_mc = Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_optimizer_mc = Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator2d_optimizer_mc = Adam(learning_rate=0.0002, beta_1=0.5)
    loss_be = tf.keras.losses.BinaryCrossentropy()
    if gan_choice == 'cWGAN':
        # Defining the model components, print_summary=True shows the model component summaries.
        d_model = get_discriminator_model(signal_length, num_classes, print_summary=True)
        g_model = get_generator_model(noise_dim, num_classes, print_summary=True)
        # Instantiate the WGAN model.
        gan = cWGAN(
            discriminator=d_model,
            generator=g_model,
            latent_dim=noise_dim,
            discriminator_extra_steps=5,
        )
    
        # Compile the cWGAN model.
        gan.compile(
            d_optimizer=discriminator_optimizer,
            g_optimizer=generator_optimizer,
            d_loss_fn=discriminator_loss,
            g_loss_fn=generator_loss
        )
    elif gan_choice == 'cDVGAN':
        # Defining the model components, print_summary=True shows the model component summaries.
        d_model = get_discriminator_model(signal_length, num_classes, print_summary=True)
        g_model = get_generator_model(noise_dim, num_classes, print_summary=True)
        d_model_derivatives = get_derivative_discriminator_model(deriv_signal_length, num_classes, print_summary=True)
        # Instantiate the DVGAN model.
        gan = cDVGAN(
            discriminator=d_model,
            deriv_discriminator = d_model_derivatives,
            generator=g_model,
            latent_dim=noise_dim,
            discriminator_extra_steps=5,
        )
    
        # Compile the DVGAN model.
        gan.compile(
            d_optimizer=discriminator_optimizer,
            d2d_optimizer=discriminator2d_optimizer,
            g_optimizer=generator_optimizer,
            d_loss_fn=discriminator_loss,
            g_loss_fn=generator_loss
        )
    elif gan_choice == 'cDVGAN2':
        d_model = get_discriminator_model(signal_length, num_classes, print_summary=True)
        g_model = get_generator_model(noise_dim, num_classes, print_summary=True)
        d_model_derivatives = get_derivative_discriminator_model(deriv_signal_length, num_classes, print_summary=True)
        d_model_derivatives2 = get_second_derivative_discriminator_model(deriv2_signal_length, num_classes, print_summary=True)
        # Instantiate the DVGAN model.
        gan = cDVGAN2(
            discriminator=d_model,
            deriv_discriminator = d_model_derivatives,
            deriv2_discriminator = d_model_derivatives2,
            generator=g_model,
            latent_dim=noise_dim,
            discriminator_extra_steps=5,
        )
    
        # Compile the DVGAN model.
        gan.compile(
            d_optimizer=discriminator_optimizer,
            d2d_optimizer=discriminator2d_optimizer,
            d2d2_optimizer=discriminator2d2_optimizer,
            g_optimizer=generator_optimizer,
            d_loss_fn=discriminator_loss,
            g_loss_fn=generator_loss
        )
    
    elif gan_choice == 'MCGANN':
        # Defining the model components, print_summary=True shows the model component summaries.
        d_model = get_discriminator_model_mc(signal_length, num_classes, print_summary=True)
        g_model = get_generator_model_mc(noise_dim, num_classes, print_summary=True)
        # Instantiate the DVGAN model.
        gan = MCGANN(
            discriminator=d_model,
            generator=g_model,
            latent_dim=noise_dim,
            discriminator_extra_steps=3,
        )
    
        # Compile the DVGAN model.
        gan.compile(
            d_optimizer=discriminator_optimizer_mc,
            g_optimizer=generator_optimizer_mc,
            d_loss_fn=loss_be,
            g_loss_fn=loss_be
        )
    elif gan_choice == 'MCDVGANN':
        # Defining the model components, print_summary=True shows the model component summaries.
        d_model = get_discriminator_model_mc(signal_length, num_classes, print_summary=True)
        g_model = get_generator_model_mc(noise_dim, num_classes, print_summary=True)
        d_model_derivatives = get_derivative_discriminator_model_mc(deriv_signal_length, num_classes, print_summary=True)
        # Instantiate the DVGAN model.
        gan = MCDVGANN(
            discriminator=d_model,
            deriv_discriminator = d_model_derivatives,
            generator=g_model,
            latent_dim=noise_dim,
            discriminator_extra_steps=1,
        )
    
        # Compile the DVGAN model.
        gan.compile(
            d_optimizer=discriminator_optimizer_mc,
            d2d_optimizer=discriminator2d_optimizer_mc,
            g_optimizer=generator_optimizer_mc,
            d_loss_fn=loss_be,
            g_loss_fn=loss_be
        )
    else:
        raise ValueError('Please choose between WGAN (1) or DVGAN (2)')


    return gan