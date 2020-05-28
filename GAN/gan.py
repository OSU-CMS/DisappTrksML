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

#import data
dataDir = '/home/MilliQan/data/disappearingTracks/tracks/'
workDir = '/home/llavezzo/'
plotDir = workDir + 'plots/images/gan_electrons/'
weightsDir = workDir + 'weights/gan_electrons/'
# dataDir = 'c:/users/llave/Documents/CMS/'
# workDir = dataDir
# plotDir = workDir + 'plots/gan/'
# weightsDir = workDir + 'weights/gan/'

# fname = 'DYJets50_norm_20x20.npy'
# data = np.load(dataDir+fname)
data_e = np.load(dataDir+'e_DYJets50_norm_20x20.npy')
data_bkg = np.load(dataDir+'bkg_DYJets50_norm_20x20.npy')
classes = np.concatenate([np.ones(len(data_e)),np.zeros(len(data_bkg))])
data = np.concatenate([data_e,data_bkg])
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]

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
def save_imgs(generator, epoch, batch, r):
    noise = np.random.normal(0, 1, (r, 100))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, 3)
    for i in range(r):
        for j in range(3):
            axs[i, j].imshow(gen_imgs[i, :, :, j], cmap='gray')
            axs[i, j].axis('off')
            axs[i,0].set_title("ECAL",fontsize=5)
            axs[i,1].set_title("HCAL",fontsize=5)
            axs[i,2].set_title("Muon",fontsize=5)
    plt.tight_layout()
    fig.savefig(plotDir+"gan_%d_%d.png" % (epoch, batch))
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

import numpy.random
from numpy.random import choice
from numpy.random import randn

def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape((n_samples, latent_dim))
    return x_input

def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)

def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3

X_train = data

epochs=100
batch_size=16
save_interval=1

num_examples = X_train.shape[0]
num_batches = int(num_examples / float(batch_size))
print('Number of examples: ', num_examples)
print('Number of Batches: ', num_batches)
print('Number of epochs: ', epochs)

half_batch = int(batch_size / 2)

d_loss = [10,10]
g_loss = 10
d_loss_array = []
g_loss_array = []

for epoch in range(epochs + 1):
    for batch in range(num_batches):

        # noise images for the batch
        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_images = generator.predict(noise)
        fake_labels = np.zeros((half_batch, 1))

        # real images for batch
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_images = X_train[idx]
        real_labels = np.ones((half_batch, 1))
        
        #testing noisy labels
        real_labels = noisy_labels(real_labels,0.05)
        real_labels = smooth_positive_labels(real_labels)
        fake_labels = smooth_negative_labels(fake_labels)

        # Train the discriminator (real classified as 1 and generated as 0)
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        labels = np.ones((batch_size, 1))

        noise = np.random.normal(0, 1, (batch_size, 100))
        #testing gaussian latent space
        noise = generate_latent_points(100,batch_size)
        
        g_loss = combined.train_on_batch(noise, labels)
            
        d_loss_array.append(d_loss[0])
        g_loss_array.append(g_loss)

        # Plot the progress
        if(batch % 50 == 0):
            #print("Epoch %d Batch %d/%d [D loss: %.2f, acc avg: %.2f%%] [D acc real: %.2f D acc fake: %.2f], [G loss: %.2f]" %
            #      (epoch, batch, num_batches, d_loss[0], 100 * d_loss[1], d_loss_real[1], d_loss_fake[1],g_loss))
             print("Epoch %d Batch %d/%d [D loss: %.2f, mse avg: %.2f] [D mse real: %.2f D mse fake: %.2f], [G loss: %.2f]" %
                  (epoch, batch, num_batches, d_loss[0], d_loss[1], d_loss_real[1], d_loss_fake[1],g_loss))
    
    
    save_imgs(generator, epoch, batch, 4)
    combined.save_weights(weightsDir+'G_epoch{0}.h5'.format(epoch))
    discriminator.save_weights(weightsDir+'D_epoch{0}.h5'.format(epoch))

plt.plot(d_loss_array,label = 'D Loss')
plt.plot(g_loss_array,label = 'G Loss')
plt.savefig(plotDir+'d_g_loss.png')
            
