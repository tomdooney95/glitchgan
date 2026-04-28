"""TensorFlow/Keras model components for cDVGAN.

Architecture is identical to the original TF reference implementation.
Lambda layers have been replaced with proper subclassed layers for
compatibility with Keras 3 and correct model serialisation.
"""

import tensorflow as tf
import keras
from keras import layers

NUM_CLASSES = 7


# ---------------------------------------------------------------------------
# Helper layers (replace deprecated Lambda layers)
# ---------------------------------------------------------------------------

class ArgmaxLayer(keras.layers.Layer):
    """Return the argmax along the last axis as int32."""
    def call(self, x):
        return tf.cast(tf.argmax(x, axis=-1), tf.int32)


class ReduceSumDotLayer(keras.layers.Layer):
    """Compute element-wise product and sum along axis 1 (keepdims)."""
    def call(self, inputs):
        x, y = inputs
        return tf.reduce_sum(x * y, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Shared conv block
# ---------------------------------------------------------------------------

def _conv_block(x, filters, activation, kernel_size=3, strides=1, padding="same",
                use_bias=True, use_bn=False, use_dropout=False, drop_value=0.5):
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding,
                      use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


# ---------------------------------------------------------------------------
# Discriminator (signal)
# ---------------------------------------------------------------------------

def get_discriminator_model(in_shape=8192, num_classes=NUM_CLASSES, print_summary=False):
    img_input = layers.Input(shape=(in_shape,))
    class_input = layers.Input(shape=(num_classes,))

    x = layers.Reshape((64, 128))(img_input)
    x = _conv_block(x, 128, kernel_size=14, strides=1, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 128, kernel_size=14, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 256, kernel_size=14, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 256, kernel_size=14, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 512, kernel_size=14, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False)(x)

    class_ind = ArgmaxLayer()(class_input)
    class_embedding = layers.Embedding(num_classes, 128)(class_ind)

    dot_product = ReduceSumDotLayer()([x, class_embedding])
    scalar_function = layers.Dense(1, use_bias=False)(x)
    out = layers.Add()([dot_product, scalar_function])

    model = keras.Model([img_input, class_input], out, name="discriminator")
    if print_summary:
        model.summary()
    return model


# ---------------------------------------------------------------------------
# Derivative discriminator (first derivative)
# ---------------------------------------------------------------------------

def get_derivative_discriminator_model(in_shape=8191, num_classes=NUM_CLASSES,
                                       print_summary=False):
    img_input = layers.Input(shape=(in_shape,))
    class_input = layers.Input(shape=(num_classes,))

    x = layers.Dense(256)(img_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 32))(x)
    x = _conv_block(x, 64, kernel_size=5, strides=1, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 128, kernel_size=5, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 256, kernel_size=5, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 256, kernel_size=5, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False)(x)

    class_ind = ArgmaxLayer()(class_input)
    class_embedding = layers.Embedding(num_classes, 128)(class_ind)

    dot_product = ReduceSumDotLayer()([x, class_embedding])
    scalar_function = layers.Dense(1, use_bias=False)(x)
    out = layers.Add()([dot_product, scalar_function])

    model = keras.Model([img_input, class_input], out, name="derivative_discriminator")
    if print_summary:
        model.summary()
    return model


# ---------------------------------------------------------------------------
# Derivative discriminator (second derivative)
# ---------------------------------------------------------------------------

def get_second_derivative_discriminator_model(in_shape=8190, num_classes=NUM_CLASSES,
                                              print_summary=False):
    img_input = layers.Input(shape=(in_shape,))
    class_input = layers.Input(shape=(num_classes,))

    x = layers.Dense(512)(img_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((32, 16))(x)
    x = _conv_block(x, 64, kernel_size=5, strides=1, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 128, kernel_size=5, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 256, kernel_size=5, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)
    x = _conv_block(x, 256, kernel_size=5, strides=2, activation=layers.LeakyReLU(),
                    use_dropout=True, drop_value=0.5)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False)(x)

    class_ind = ArgmaxLayer()(class_input)
    class_embedding = layers.Embedding(num_classes, 128)(class_ind)

    dot_product = ReduceSumDotLayer()([x, class_embedding])
    scalar_function = layers.Dense(1, use_bias=False)(x)
    out = layers.Add()([dot_product, scalar_function])

    model = keras.Model([img_input, class_input], out,
                        name="second_derivative_discriminator")
    if print_summary:
        model.summary()
    return model


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _upsample_block(x, filters, activation, kernel_size=18, up_size=2,
                    padding="same", use_bn=False, use_bias=True,
                    use_dropout=False, drop_value=0.3):
    x = layers.UpSampling1D(up_size)(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding=padding,
                      use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_generator_model(noise_dim=100, num_classes=NUM_CLASSES, print_summary=False):
    noise = layers.Input(shape=(noise_dim,))
    class_input = layers.Input(shape=(num_classes,))

    class_embedding = layers.Dense(32, use_bias=False)(class_input)
    combined = layers.Concatenate()([noise, class_embedding])

    x = layers.Dense(4096, use_bias=False)(combined)
    x = layers.ReLU()(x)
    x = layers.Reshape((256, 16))(x)

    x = _upsample_block(x, 512, layers.ReLU(), use_bias=False, use_bn=True)
    x = _upsample_block(x, 256, layers.ReLU(), use_bias=False, use_bn=True)
    x = _upsample_block(x, 128, layers.ReLU(), use_bias=False, use_bn=True)
    x = _upsample_block(x, 64,  layers.ReLU(), use_bias=False, use_bn=True)
    x = _upsample_block(x, 1, layers.Activation("linear"), use_bias=False, use_bn=False)

    x = layers.Flatten()(x)

    model = keras.Model([noise, class_input], x, name="generator")
    if print_summary:
        model.summary()
    return model
