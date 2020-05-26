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
plotDir = workDir + 'images/gan/'
weightsDir = workDir + 'weights/gan/'

numImages = 4
epoch = 20

def build_discriminator(img_shape):
    input = Input(img_shape)
    x = Conv2D(32*3, kernel_size=(4,4), strides=(2,2), padding="same")(input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64*3, kernel_size=(4,4), strides=(2,2), padding="same")(x)
    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
    x = (LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(128*3, kernel_size=(4,4), strides=(2,2), padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(256*3, kernel_size=(4,4), strides=(1,1), padding="same")(x)
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
    x = Dense(128 * 5 * 5, activation="relu")(input)
    x = Reshape((5,5, 128))(x)
    x = BatchNormalization(momentum=0.8)(x)
    #upsampling to 20x20
    x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    #upsampling to 40x40
    x = Conv2DTranspose(64, (4,4),strides=(2,2), padding='same')(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(3, kernel_size=3, padding="same")(x)
    out = Activation("relu")(x)
    model = Model(input, out)
    print("-- Generator -- ")
    model.summary()
    return model

#generates and saves r random images
def save_imgs(generator, epoch, r):
    noise = np.random.normal(0, 1, (r, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, 3,figsize=(10,10))
    for i in range(r):
        for j in range(3):
            axs[i, j].imshow(gen_imgs[i, :, :, j], cmap='gray')
            axs[i, j].axis('off')
            axs[i,0].set_title("ECAL",fontsize=5)
            axs[i,1].set_title("HCAL",fontsize=5)
            axs[i,2].set_title("Muon",fontsize=5)
    plt.tight_layout()
    fig.savefig(plotDir+"gen_%d.png" % (epoch))
    plt.close()

#build and compile discriminator and generator
discriminator = build_discriminator(img_shape=(20,20, 3))
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


combined.load_weights(weightsDir+'G_epoch{0}.h5'.format(epoch))
save_imgs(generator, epoch, numImages)
            

