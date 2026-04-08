# Imports.
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import RMSprop
import numpy as np

from gan_models import choose_gan
from utils import discriminator_loss, generator_loss, calculate_derivative, plot_GAN_history, plot_examples, GANMonitor, fit_GAN, Examples_generator

import pickle
import json

print(tf.config.list_physical_devices('GPU'))

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("GPU is available and should be used.")
else:
    print("GPU is not available. TensorFlow might be running on CPU.")

# List all available GPUs
gpus = tf.config.list_physical_devices('GPU')

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", gpus)

if gpus:
    # Select the last available GPU
    last_gpu = gpus[0]
    
    # Restrict TensorFlow to only use the last GPU
    tf.config.set_visible_devices(last_gpu, 'GPU')
    
    # Verify that only the last GPU is being used
    visible_devices = tf.config.get_visible_devices('GPU')
    print("Using GPU:", visible_devices)
else:
    print("GPU is not available. TensorFlow might be running on CPU.")

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set parameters.
sample_rate = 4096
noise_dim = 100
num_classes = 2

# Set directories for storing outputs
output_dir = 'GAN_outputs/'
monitor_dir = 'Monitor/'

data = np.load('data/glitch_GAN_samples_scaled.npy')
data_deriv = np.load('data/glitch_GAN_deriv_samples.npy')
class_array = np.load('data/glitch_GAN_labels.npy')

data_deriv2 = None

gan_choice_dict = {1:'cWGAN',
                  2:'cDVGAN',
                  3:'cDVGAN2',
                  4:'MCGANN',
                  5:'MCDVGANN'}
# gan_choice_int = int(input('Please input the GAN variant to train (1: cWGAN, 2: cDVGAN, 3: cDVGAN2, 4: MCGANN, 5: MCDVGANN): '))
# gan_choice = gan_choice_dict[gan_choice_int]

gan_choice = 'cDVGAN'

# Change the below to 'WGAN' to train a vanilla model
signal_length = data.shape[-1]
deriv_signal_length = data_deriv.shape[-1]
deriv2_signal_length = None

# Create and compile GAN model
gan = choose_gan(gan_choice, signal_length, deriv_signal_length, deriv2_signal_length, num_classes, noise_dim)

# Set batch size and number of epochs for training.
BATCH_SIZE = 2
epochs = 500
# Change to False if you don't want the GAN monitor to plot generated signals after each epoch.
callback = True
# Callback path to save GAN monitor signals.
callback_path = monitor_dir+gan_choice+'/'

if gan_choice in ['cDVGAN', 'MCDVGANN']:
    data = [data, data_deriv, class_array]
# elif gan_choice == 'cDVGAN2':
#     data = [data, data_deriv, data_deriv2, class_array]
else:
    data = [data, class_array]
    
# Start training the model, saving the histroy information.
history = fit_GAN(gan, data, batch_size=BATCH_SIZE, epochs=epochs, gan_variant = gan_choice, callback = callback, noise_dim = noise_dim, callback_path=callback_path)

# Output path where loss plots, generated examples and trained generators are stored.
output_path = output_dir+gan_choice
isExist = os.path.exists(output_path)
if not isExist:
    os.makedirs(output_path)

# Plot training history.
plot_GAN_history(history, output_path, gan_choice)

# Save history as json.
# Get the dictionary containing each metric and the loss for each epoch.
history_dict = history.history
# Dump it
json.dump(history_dict, open(output_path+'/history.json', 'w'))

# Save all components
gan.generator.save(output_path+'/Generator.keras')
gan.discriminator.save(output_path+'/Discriminator.keras')

# If using DVGAN, save additional models
if gan_choice in ['cDVGAN', 'cDVGAN2', 'MCDVGANN']:
    gan.deriv_discriminator.save(output_path+'/Deriv_Discriminator.keras')
    if gan_choice == 'cDVGAN2':
        gan.deriv2_discriminator.save(output_path+'/Deriv2_Discriminator.keras')

        


# Generate signals. We will sample the class space in three different ways; vertex, simplex and uniform sampling
num_signals = 10
indices = tf.experimental.numpy.random.randint(
        0,
        high=num_classes,
        size=[num_signals])
depth = num_classes

vertex_classes = tf.one_hot(indices, depth,
          on_value=1.0, off_value=0.0,
          axis=-1).numpy()

random_ints = np.random.randint(0, 100, size=(num_signals,num_classes))

simplex_classes = random_ints/np.sum(random_ints, axis=1).reshape(num_signals,1)

uniform_classes = np.random.uniform(low=0.0, high=1.0, size=(num_signals,num_classes))


latent_vectors_vertex = tf.random.normal(shape=(num_signals, noise_dim)).numpy()
latent_vectors_simplex = tf.random.normal(shape=(num_signals, noise_dim)).numpy()
latent_vectors_uniform = tf.random.normal(shape=(num_signals, noise_dim)).numpy()

generations_vertex = gan.generator([latent_vectors_vertex, vertex_classes])
generations_vertex = generations_vertex.numpy()

generations_simplex = gan.generator([latent_vectors_vertex, simplex_classes])
generations_simplex = generations_simplex.numpy()

generations_uniform = gan.generator([latent_vectors_vertex, uniform_classes])
generations_uniform = generations_uniform.numpy()


# Plot some examples of generated data using different sampling methods.
plot_examples(generations_vertex, vertex_classes, output_path+'/Vertex_examples')
plot_examples(generations_simplex, simplex_classes, output_path+'/Simplex_examples')
plot_examples(generations_uniform, uniform_classes, output_path+'/Uniform_examples')


print('Training Completed!')