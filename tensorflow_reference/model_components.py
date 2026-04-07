import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras import backend as K
num_classes = 7
def conv_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=1,
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.5,
):
    x = layers.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_discriminator_model(in_shape=8192, num_classes=num_classes, print_summary = False):
    img_input = layers.Input(shape=(in_shape,))
    class_input = layers.Input(shape=(num_classes,))
    x = layers.Reshape((64,128))(img_input)
    x = conv_block(
        x,
        128,
        kernel_size=14,
        strides=1,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        128,
        kernel_size=14,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=14,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=14,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        512,
        kernel_size=14,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    # 
    class_ind = layers.Lambda(lambda x: tf.argmax(x, axis=-1))(class_input)
    class_embedding = layers.Embedding(num_classes, 128)(class_ind)

    # dot_product = tf.reduce_sum(tf.multiply(x, class_embedding), axis=1, keepdims=True)
    dot_product = layers.Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1]), axis=1, keepdims=True))([x, class_embedding])
    scalar_function = layers.Dense(1, use_bias=False)(x)

    x = layers.Add()([dot_product, scalar_function])
    
    d_model = keras.models.Model([img_input,class_input], x, name="discriminator")
    if print_summary:
        print(d_model.summary())
    return d_model



def get_derivative_discriminator_model(in_shape=8191, num_classes=num_classes, print_summary = False):
    img_input = layers.Input(shape=(in_shape,))
    class_input = layers.Input(shape=(num_classes,))
    x = layers.Dense(256)(img_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 32))(x)
    x = conv_block(
        x,
        64,
        kernel_size=5,
        strides=1,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        128,
        kernel_size=5,
        strides=2,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=5,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=5,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )


    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    
    class_ind = layers.Lambda(lambda x: tf.argmax(x, axis=-1))(class_input)
    class_embedding = layers.Embedding(num_classes, 128)(class_ind)

    # dot_product = tf.reduce_sum(tf.multiply(x, class_embedding), axis=1, keepdims=True)
    dot_product = layers.Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1]), axis=1, keepdims=True))([x, class_embedding])
    scalar_function = layers.Dense(1, use_bias=False)(x)

    x = layers.Add()([dot_product, scalar_function])

    d_model = keras.models.Model([img_input,class_input], x, name="derivative_discriminator")
    if print_summary:
        print(d_model.summary())
    return d_model


def get_second_derivative_discriminator_model(in_shape=1022, num_classes=num_classes, print_summary = False):
    img_input = layers.Input(shape=(in_shape,))
    class_input = layers.Input(shape=(num_classes,))
    x = layers.Dense(512)(img_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((32,16))(x)
    x = conv_block(
        x,
        64,
        kernel_size=5,
        strides=1,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        128,
        kernel_size=5,
        strides=2,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(),
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=5,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )
    x = conv_block(
        x,
        256,
        kernel_size=5,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(),
        use_bias=True,
        use_dropout=True,
        drop_value=0.5,
    )


    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    
    class_ind = layers.Lambda(lambda x: tf.argmax(x, axis=-1))(class_input)
    class_embedding = layers.Embedding(num_classes, 128)(class_ind)

    # dot_product = tf.reduce_sum(tf.multiply(x, class_embedding), axis=1, keepdims=True)
    dot_product = layers.Lambda(lambda x: tf.reduce_sum(tf.multiply(x[0], x[1]), axis=1, keepdims=True))([x, class_embedding])
    scalar_function = layers.Dense(1, use_bias=False)(x)

    x = layers.Add()([dot_product, scalar_function])

    d_model = keras.models.Model([img_input,class_input], x, name="derivative2_discriminator")
    if print_summary:
        print(d_model.summary())
    return d_model

def upsample_block(
    x,
    filters,
    activation,
    kernel_size=18,
    strides=2,
    up_size=2,
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=0.3,
):
    x = layers.UpSampling1D(up_size)(x)
    x = layers.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


 
def get_generator_model(noise_dim=100, num_classes=num_classes, print_summary = False):
    noise = layers.Input(shape=(noise_dim,))
    class_input = layers.Input(shape=(num_classes,))
    # class_ind = K.argmax(class_input, axis=-1)
    # class_embedding = layers.Embedding(num_classes, 32)(class_ind)
    class_embedding = layers.Dense(32, use_bias=False)(class_input)
    combined_input = layers.Concatenate()([noise,class_embedding])
    x = layers.Dense(4096, use_bias=False)(combined_input)
    x = layers.ReLU()(x)

    x = layers.Reshape((256, 16))(x)
    x = upsample_block(
        x,
        512,
        layers.ReLU(),
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        256,
        layers.ReLU(),
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        128,
        layers.ReLU(),
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        64,
        layers.ReLU(),
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )

    x = upsample_block(
        x,
        1,
        layers.Activation('linear'),
        strides=1,
        use_bias=False,
        use_bn=False,
        padding="same",
        use_dropout=False,
    )
    x = layers.Flatten()(x)

    g_model = keras.models.Model([noise, class_input], x, name="generator")
    if print_summary:
        print(g_model.summary())
    return g_model


def get_discriminator_model_mc(in_shape=1024,n_classes=3, print_summary = False):
    # i'm using functional API since its more flexible and can handle mulitple inputs better (not a sequential model).
    # label input
    in_label = layers.Input(shape=(n_classes,))
    # scale up ti image dim with linear activation (linear also means no activation in this case)
    n_nodes = in_shape
    li = layers.Dense(n_nodes)(in_label)
    # reshape to additional channel
    li = layers.Reshape((in_shape,1))(li)
    # image input
    in_image = layers.Input(shape=(1024,))
    In = layers.Reshape((in_shape,1))(in_image)
    # concat label as a channel!
    merge = layers.Concatenate()([In, li])
    # downsample
    fe = layers.Conv1D(64, 14, strides=2, padding='same')(merge)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(128, 14, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(256, 14, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(512, 14, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    # flatten feature map
    fe = layers.Flatten()(fe)
    # Dropout
    #fe = Dropout(0.5)(fe)
    # output
    out_layer = layers.Dense(1, activation='sigmoid')(fe)
    # define model
    model = keras.models.Model([in_image, in_label], out_layer, name="discriminator_mc")
    # complie that mofo
    # opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.summary()
    if print_summary:
        print(model.summary())
    return model

def get_derivative_discriminator_model_mc(in_shape=1023,n_classes=3, print_summary = False):
    # i'm using functional API since its more flexible and can handle mulitple inputs better (not a sequential model).
    # label input
    in_label = layers.Input(shape=(n_classes,))
    # scale up ti image dim with linear activation (linear also means no activation in this case)
    n_nodes = in_shape
    li = layers.Dense(n_nodes)(in_label)
    # reshape to additional channel
    li = layers.Reshape((in_shape,1))(li)
    # image input
    in_image = layers.Input(shape=(in_shape,))
    In = layers.Reshape((in_shape,1))(in_image)
    # concat label as a channel!
    merge = layers.Concatenate()([In, li])
    # downsample
    fe = layers.Conv1D(64, 14, strides=2, padding='same')(merge)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(128, 14, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(256, 14, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    fe = layers.Conv1D(512, 14, strides=2, padding='same')(fe)
    fe = layers.LeakyReLU(alpha=0.2)(fe)
    fe = layers.SpatialDropout1D(0.5)(fe)

    # flatten feature map
    fe = layers.Flatten()(fe)
    # Dropout
    #fe = Dropout(0.5)(fe)
    # output
    out_layer = layers.Dense(1, activation='sigmoid')(fe)
    # define model
    model = keras.models.Model([in_image, in_label], out_layer, name="derivative_discriminator_mc")
    # complie that mofo
    # opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    if print_summary:
        print(model.summary())
    return model


def get_generator_model_mc(latent_dim = 100, n_classes=3, print_summary = False):
    # image input
    in_lat = layers.Input(shape=(latent_dim,))
    in_label = layers.Input(shape=(n_classes,))

    x = layers.Concatenate()([in_lat, in_label])

    n_nodes = 64 * 512
    merge = layers.Dense(n_nodes)(x)
    merge = layers.Activation('relu')(merge)

    # Add dimension as there's no 1DTranspose
    merge = layers.Reshape((64,1,512))(merge)
    # upsample
    #gen = Conv2DTranspose(512, kernel_size=(18,1), strides=(1,1), padding='same')(merge)
    #gen = Activation('relu')(gen)
    #gen = BatchNormalization()(gen)

    gen = layers.Conv2DTranspose(256, kernel_size=(18,1), strides=(2,1), padding='same')(merge)
    gen = layers.Activation('relu')(gen)

    gen = layers.Conv2DTranspose(128, kernel_size=(18,1), strides=(2,1), padding='same')(gen)
    gen = layers.Activation('relu')(gen)

    gen = layers.Conv2DTranspose(64, kernel_size=(18,1), strides=(2,1), padding='same')(gen)
    gen = layers.Activation('relu')(gen)

    gen = layers.Conv2DTranspose(1, kernel_size=(18,1), strides=(2,1), padding='same')(gen)
    gen = layers.Activation('linear')(gen)
   # output
    out_layer = layers.Reshape((1024,))(gen)
    # define model
    model = keras.models.Model([in_lat, in_label], [out_layer], name="generator_mc")
    if print_summary:
        print(model.summary())
    return model

