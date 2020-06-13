import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

#set directories
workDir = "/Users/czkaiweb/Research/DisappTrksML/VAE/"
saveDir = workDir
dataDir = "/Users/czkaiweb/Research/DisappTrkMLData/converted/"

#load data
df = pd.read_pickle(dataDir+'images_DYJets50_norm_40x40_tanh.pkl')


df_electron_list = df.loc[df['type']==1.0]
df_muon_list     = df.loc[df['type']==2.0]
df_bkg_list      = df.loc[df['type']==0.0]

# For GEN-matched sub-channel
df_electron = df_electron_list.iloc[:,4:].to_numpy()
df_muon     = df_muon_list.iloc[:,4:].to_numpy()
df_bkg      = df_bkg_list.iloc[:,4:].to_numpy()

print "nElectron:",(df_electron.shape[0])
print "nMuon:"(df_muon.shape[0])
print "nOther:"(df_bkg.shape[0])

# Get numpy
x = df.to_numpy()[:,4:]
y = df.to_numpy()[:,0]

input_train, input_test, target_train,target_test = train_test_split(x, y, test_size=0.30, random_state=42)

input_train = input_train.astype('float32')
input_test = input_test.astype('float32')
target_train = target_train.astype('int64')
target_test = target_test.astype('int64')

# Data & model configuration
img_width, img_height = 40,40
num_channels = 4

# Reshape data
input_train = input_train.reshape(input_train.shape[0], img_height, img_width, num_channels)
input_test = input_test.reshape(input_test.shape[0], img_height, img_width, num_channels)


# Layer selection
Layer = "ECAL"
LayerList=["ECAL","HCAL","MUON"]
iLayer = LayerList.index(Layer)
input_train = input_train[:,:,:,[iLayer]]
input_test = input_test[:,:,:,[iLayer]]
num_channels = 1
input_shape = (img_height, img_width, num_channels)

# # =================
# # Encoder
# # =================

# Definition
i       = Input(shape=input_shape, name='encoder_input')
cx      = Conv2D(filters=num_channels*4, kernel_size=3, strides=2, padding='same', activation='relu')(i)
#cx      = BatchNormalization()(cx)
#cx      = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
#cx      = BatchNormalization()(cx)
x       = Flatten()(cx)
x       = Dense(20, activation='relu')(x)
#x       = BatchNormalization()(x)
mu      = Dense(latent_dim, name='latent_mu')(x)
sigma   = Dense(latent_dim, name='latent_sigma')(x)

# Get Conv2D shape for Conv2DTranspose operation in decoder
conv_shape = K.int_shape(cx)

# Define sampling with reparameterization trick
def sample_z(args):
    mu, sigma = args
    batch     = K.shape(mu)[0]
    dim       = K.int_shape(mu)[1]
    eps       = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps

# Use reparameterization trick to ....??
z  = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])

# Instantiate encoder
encoder = Model(i, [mu, sigma, z], name='encoder')
encoder.summary()

# =================
# Decoder
# =================

# Definition
d_i   = Input(shape=(latent_dim, ), name='decoder_input')
x     = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
#x     = BatchNormalization()(x)
x     = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
cx    = Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding='same', activation='relu')(x)
#cx    = BatchNormalization()(cx)
#cx    = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
#cx    = BatchNormalization()(cx)
#o     = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
o     = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='relu', padding='same', name='decoder_output')(cx)

# Instantiate decoder
decoder = Model(d_i, o, name='decoder')
decoder.summary()

# =================
# VAE as a whole
# =================

# Instantiate VAE
vae_outputs = decoder(encoder(i)[2])
vae         = Model(i, vae_outputs, name='vae')
vae.summary()

# Define loss
def kl_reconstruction_loss(true, pred):
    # Reconstruction loss
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height * num_channels
    # KL divergence loss
    kl_loss = 1 + sigma  - K.exp(sigma) - K.square(mu)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + 0.001*kl_loss)

# Compile VAE
vae.compile(optimizer='adam', loss=kl_reconstruction_loss)
#vae.compile(optimizer='adadelta', loss=kl_reconstruction_loss)
#vae.compile(optimizer='rmsprop', loss=kl_reconstruction_loss)

# Train autoencoder
history = vae.fit(input_train, input_train, epochs = no_epochs, batch_size = batch_size, validation_split = validation_split)


