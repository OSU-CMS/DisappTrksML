from keras import models
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from itertools import repeat
import os

dataDir = "/data/disappearingTracks/muon_selection_DYJetsToll_M50/"
weightsDir = "/home/llavezzo/weights/ae/"
plotDir = "/home/llavezzo/plots/ae/"

#how many bkg events to train on
nBkgTot = 15000
val_size = 0.3

with open(dataDir+'mSignalCounts.json') as json_file:
    mCounts = json.load(json_file)
with open(dataDir+'mBackgroundCounts.json') as json_file:
    bkgCounts = json.load(json_file)

files = []
for key, value in mCounts.items():
    if(value > 0): files.append(key)

bkg_images = []
i=0
while nBkg < nBkgTot*(1-val_size):
    if(str(i) in files): continue
    fname = "bkg_0p25_tanh_"+str(i)+".npz"
    temp = np.load(dataDir+fname)
    bkg_images.append(temp['images'])
    nBkgTot+=bkg_images.shape[0]
    i+=1
bkg_images = np.vstack(bkg_images)
bkg_images = bkg_images[:,1:]
bkg_images = np.reshape(bkg_images,(bkg_images.shape[0],40,40,4))
bkg_images = bkg_images[:,:,:,[3]]

print("Training on",bkg_images.shape[0],"background images")

x_train, x_test = train_test_split(bkg_images, val_size=val_size, random_state=42)

encoder = models.Sequential()

encoder.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',input_shape=(40,40,1)))
encoder.add(layers.MaxPooling2D((2,2), padding='same'))
encoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
encoder.add(layers.MaxPooling2D((2,2), padding='same'))
encoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
encoder.add(layers.MaxPooling2D((2,2), padding='same'))
print("encoder===>")
print(encoder.summary())

decoder = models.Sequential()
decoder.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same',input_shape=(5,5,16)))
decoder.add(layers.UpSampling2D((2,2)))
decoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
decoder.add(layers.UpSampling2D((2,2)))
decoder.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
decoder.add(layers.UpSampling2D((2,2)))
decoder.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
print("decoder===>")
print(decoder.summary())


autoencoder = models.Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['mse'])
print("autoencoder===>")
print(autoencoder.summary())

if(True):
    history = autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=64,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x_test_reduced, x_test_reduced))

    autoencoder.save_weights(weightsDir+'ae_first_model.h5')
else:
    autoencoder.load_weights(weightsDir+'ae_first_model.h5')


for i,file in enumerate(files):
    fname = "m_0p25_tanh_"+file+".npz"
    fname2 = "bkg_0p25_tanh_"+file+".npz"
    temp = np.load(dataDir+fname)
    temp2 = np.load(dataDir+fname2)
    m_images = temp[:,1:]
    m_images = np.reshape(m_images,(m_images.shape[0],40,40,4))
    m_images = m_images[:,:,:,[3]]
    bkg_images = temp2[:,1:]
    bkg_images = np.reshape(bkg_images,(bkg_images.shape[0],40,40,4))
    bkg_images = bkg_images[:,:,:,[3]]

    decoded_m_imgs = autoencoder.predict(m_images)
    decoded_bkg_imgs = autoencoder.predict(bkg_images)

    decoded_bkg_imgs_flat = np.reshape(decoded_bkg_imgs, (decoded_bkg_imgs.shape[0],40*40))
    decoded_m_imgs_flat = np.reshape(decoded_m_imgs, (decoded_m_imgs.shape[0],40*40))
    bkg_images_flat = np.reshape(bkg_images, (bkg_images.shape[0],40*40))
    m_images_flat = np.reshape(m_images, (m_images.shape[0],40*40))

    if(i==0):
        mse_bkg = np.mean(np.power(bkg_images_flat-decoded_bkg_imgs_flat, 2), axis=1)
        mse_m = np.mean(np.power(m_images_flat-decoded_m_imgs_flat, 2), axis=1)
    else:
        mse_bkg = np.concatenate((mse_bkg,  np.mean(np.power(bkg_images_flat-decoded_bkg_imgs_flat, 2), axis=1)))
        mse_m = np.concatenate((mse_m,  np.mean(np.power(m_images_flat-decoded_m_imgs_flat, 2), axis=1)))


np.save(plotDir+"mse_bkg.npy",mse_bkg)
np.save(plotDir+"mse_muons.npy",mse_m)

plt.hist(mse_bkg,label="bkg",alpha=0.5,bins=50)
plt.hist(mse_m,label="m",alpha=0.5,bins=50)
plt.legend()
plt.yscale("log")
plt.savefig(plotDir+"mse.png")