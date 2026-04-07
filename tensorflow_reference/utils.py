import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import backend as K
import os
import numpy as np

num_classes = 7
output_dir = 'Monitor/'
output_path = output_dir



# Define the loss function for the discriminators,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_sig, fake_sig):
    """Calculate the Wasserstein Loss for discriminators.
    
    Parameters
    ---------

    real_sig: class: 'tf.Tensor'
            logits for real signals
            
    fake_sig: class: 'tf.Tensor'
            logits for fake signals
            
    Returns
    ---------
    
    Discriminator Wasserstein loss
    """
    real_loss = tf.reduce_mean(real_sig)
    fake_loss = tf.reduce_mean(fake_sig)
    return fake_loss - real_loss


# Define the loss function for the generator.
def generator_loss(fake_sig):
    """Calculate the Wasserstein Loss for generator.
    
    Parameters
    ---------     
    
    fake_sig: class: 'tf.Tensor'
            logits for fake signals
            
    Returns
    ---------
    
    Generator Wasserstein loss
    """
    return -tf.reduce_mean(fake_sig)

@tf.function
def calculate_derivative(x, y):
    """A tf function that calculates the derivative of one vector with respect to another.
    Used to calculate derivative signals of generated samples during training."""
    dydx = tf.experimental.numpy.diff(y)/tf.experimental.numpy.diff(x)
    return dydx

    
def plot_GAN_history(history, path, gan_variant):
    """A function to plot and save the WGAN or DVGAN losses to a certain path."""
    if gan_variant in ['cDVGAN', 'cDVGAN2', 'MCDVGANN']:
        plt.figure()
        plt.plot(history.history['d_loss'], 'b-', label = 'Discriminator 1')
        plt.plot(history.history['d2d_loss'], 'r-', label = 'Discriminator 2')
        plt.plot(history.history['g_loss'], 'g-', label = 'Generator 1')
        plt.plot(history.history['g_loss2d'], 'c-', label = 'Generator 2')
        plt.plot(history.history['g_loss_combined'], 'm-', label = 'Generator Combined')
        if gan_variant == 'cDVGAN2':
            plt.plot(history.history['d2d2_loss'], 'r-', label = 'Discriminator 3')
            plt.plot(history.history['g_loss2d2'], 'c-', label = 'Generator 3')
    else:
        plt.figure()
        plt.plot(history.history['d_loss'], 'b-', label = 'Discriminator')
        plt.plot(history.history['g_loss'], 'm-', label = 'Generator')
        
    plt.legend()
    plt.savefig(path+f'/{gan_variant}_loss_plot')
    plt.close()
    

def plot_examples(data, classes, path):
    """A function to plot and save 9 examples of training or generated data."""
    plt.figure(figsize=(12,7))
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        ax.plot(data[i])
        ax.set_title(classes[i].round(3))
        plt.subplots_adjust(hspace=0.4)

    plt.savefig(path)
    plt.close()


    
def fit_GAN(GAN, data, batch_size, epochs, gan_variant = 'DVGAN', callback = False, noise_dim = 100, callback_path = 'DVGAN_monitor'):
    """A function to train the GAN model. We can choose to pass call backs to monitor the training after each epoch.
    
    Parameters
    ---------
    
    GAN: class: 'keras.Model'
            The GAN model to fit to data

    data: class:'numpy.ndarray' or 'tf.Tensor'
            A 1D array holding signals
            
    data_deriv: class:'numpy.ndarray' or 'tf.Tensor'
            A 1D array holding derivative signals
            
    batch_size: int
            The batch size for the model
        
    epochs: int
            The number of epochs to train the model
            
    gan_variant: str
            The variant of the GAN to be trained, either 'WGAN' or 'DVGAN'
          
    callback: bool
            Boolean controlling whether GAN monitor should be used
    
    callback_path: str
            The path where GAN monitor images are saved during training
            
    """

    if callback:
        # Instantiate the 'GANMonitor' for Keras callbacks.
        cbk = GANMonitor(num_img=1, latent_dim=noise_dim, gan_variant = gan_variant, callback_path=callback_path)
        isExist = os.path.exists(callback_path)
        if not isExist:
            os.makedirs(callback_path)
        
        history = GAN.fit(data, batch_size=batch_size, epochs=epochs, callbacks = [cbk])
    else:
        history = GAN.fit(data, batch_size=batch_size, epochs=epochs)
        
    return history
    
    
class GANMonitor(keras.callbacks.Callback):
    """GAN monitor used to plot GAN generated data after each epoch."""
    def __init__(self, num_img=1, latent_dim=100, gan_variant='WGAN', callback_path = 'WGAN_monitor'):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.gan_variant = gan_variant
        self.callback_path = callback_path

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
               
        indices = tf.experimental.numpy.random.randint(
                0,
                high=num_classes,
                size=[self.num_img])
        depth = num_classes
        random_classes = tf.one_hot(indices, depth,
                  on_value=1.0, off_value=0.0,
                  axis=-1)
        generated_signals = self.model.generator([random_latent_vectors, random_classes])    
        

        for i in range(self.num_img):
            img = generated_signals[i].numpy()
            plt.figure()
            plt.plot(img)
            plt.savefig(self.callback_path+"/generated_img_{i}_{random_classes}_{epoch}.png".format(i=i, random_classes=random_classes, epoch=epoch),format="png")
            plt.close()
        
        # We will also save model components every 100 epochs
        if epoch%10==0:
            
            self.model.generator.save(self.callback_path + f'generator_{epoch}.keras') 
            self.model.discriminator.save(self.callback_path+f'discriminator_{epoch}.keras')
            if self.gan_variant in ['DVGAN', 'DVGAN2', 'MCDVGANN']:
                self.model.deriv_discriminator.save(self.callback_path+f'deriv_discriminator_{epoch}.keras')
                if self.gan_variant == 'DVGAN2':
                    self.model.deriv2_discriminator.save(self.callback_path+f'deriv2_discriminator_{epoch}.keras')
            Examples_generator(self.model,epoch)
     
def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def Examples_generator(gan,epoch='last',noise_dim = 100,switch_sim_uni = False,output_path = output_path):
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
    plot_examples(generations_vertex, vertex_classes, output_path+f'/Vertex_examples_epoch{epoch}')
    if switch_sim_uni:
        plot_examples(generations_simplex, simplex_classes, output_path+f'/Simplex_examples_epoch{epoch}')
        plot_examples(generations_uniform, uniform_classes, output_path+f'/Uniform_examples_epoch{epoch}')      
   