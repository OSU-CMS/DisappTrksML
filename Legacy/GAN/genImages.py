import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

workDir = '/home/llavezzo/'
saveDir = workDir + 'gan_gen/'
weightsDir = workDir + 'weights/gan_electrons/'

numImages = 5000
load_epoch = 199

def build_discriminator(img_shape):
    input = Input(img_shape)
    x = Conv2D(32*4, kernel_size=(4,4), strides=(2,2), padding="same")(input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64*4, kernel_size=(4,4), strides=(2,2), padding="same")(x)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    x = (LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(128*4, kernel_size=(4,4), strides=(2,2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(256*4, kernel_size=(4,4), strides=(1,1), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    #testing
    out = Dense(1, activation='sigmoid')(x)

    model = Model(input, out)
    print("-- Discriminator -- ")
    model.summary()
    return model


def build_generator(noise_shape=(100,)):
    input = Input(noise_shape)
    x = Dense(128 * 10 * 10, activation="relu")(input)
    x = Reshape((10,10, 128))(x)
    x = BatchNormalization(momentum=0.8)(x)
    #upsampling to 20x20
    x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    #upsampling to 40x40
    x = Conv2DTranspose(64, (4,4),strides=(2,2), padding='same')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(4, kernel_size=3, padding="same")(x)
    out = Activation("relu")(x)
    model = Model(input, out)
    print("-- Generator -- ")
    return model

#build and compile discriminator and generator
discriminator = build_discriminator(img_shape=(40,40, 4))
discriminator.compile(loss='binary_crossentropy',
                               optimizer=Adam(lr=0.0002, beta_1=0.5),
                               metrics=['mse'])

generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

#combine them
z = Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
real = discriminator(img)
combined = Model(z, real)
combined.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape((n_samples, latent_dim))
    return x_input

combined.load_weights(weightsDir+'G_epoch{0}.h5'.format(load_epoch))

noise = np.random.normal(0, 1, (numImages, 100))
gen_images = generator.predict(noise)
assert len(gen_images) == numImages

print("Saving",numImages,"generated images in",saveDir)
np.save(saveDir + 'genImages',gen_images)