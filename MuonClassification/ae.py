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
workDir = '/home/llavezzo/'
weightsDir = workDir + 'weights/'
train = False

with open(dataDir+'mSignalCounts.json') as json_file:
    mCounts = json.load(json_file)
with open(dataDir+'mBackgroundCounts.json') as json_file:
    bkgCounts = json.load(json_file)

files = []
for key, value in mCounts.items():
    if(value > 0): files.append(key)

m_images = []
for file in files:
    fname = "m_0p25_tanh_"+file+".npz"
    temp = np.load(dataDir+fname)
    m_images.append(temp['images'])
m_images = np.vstack(m_images)
m_images = m_images[:,1:]
m_images = np.reshape(m_images,(m_images.shape[0],40,40,4))
m_images = m_images[:,:,:,[3]]
m_images = np.arctanh(m_images)

bkg_images = []
for i in range(10):
    fname = "bkg_0p25_tanh_"+str(i)+".npz"
    if(not os.path.isfile(dataDir+fname)): continue
    temp = np.load(dataDir+fname)
    bkg_images.append(temp['images'])
bkg_images = np.vstack(bkg_images)
bkg_images = bkg_images[:,1:]
bkg_images = np.reshape(bkg_images,(bkg_images.shape[0],40,40,4))
bkg_images = bkg_images[:,:,:,[3]]
bkg_images = np.arctanh(bkg_images)

numSignal = len(m_images)
numBkg = len(bkg_images)
print("Loaded",numSignal,"muon events and",numBkg,"background events")
x = bkg_images

x_train, x_test = train_test_split(x, test_size=0.2, random_state=42) 

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

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['mse'])
print("autoencoder===>")
print(autoencoder.summary())

if(False):
    history = autoencoder.fit(x_train, x_train,
                    epochs=20,
                    batch_size=32,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x_test, x_test))

    autoencoder.save_weights(weightsDir+'ae_first_model.h5')
else:
    autoencoder.load_weights(weightsDir+'ae_first_model.h5')

decoded_bkg_imgs = autoencoder.predict(x_test)
decoded_m_imgs = autoencoder.predict(m_images)

for i in range(10):
    fig, axs = plt.subplots(1,2,figsize=(10,10))
    axs[0].set_title("Muon")
    axs[1].set_title("ae - Muon")
    axs[0].imshow(x_test[i,:,:,0])
    axs[1].imshow(decoded_bkg_imgs[i,:,:,0])
    fig.savefig("bkg_"+str(i)+".png")
    fig.clf()

for i in range(10):
    fig, axs = plt.subplots(1,2,figsize=(10,10))
    axs[0].set_title("Muon")
    axs[1].set_title("ae - Muon")
    axs[0].imshow(m_images[i,:,:,0])
    axs[1].imshow(decoded_m_imgs[i,:,:,0])
    fig.savefig("m_"+str(i)+".png")
    fig.clf()

decoded_bkg_imgs_flat = np.reshape(decoded_bkg_imgs, (decoded_bkg_imgs.shape[0],40*40))
decoded_m_imgs_flat = np.reshape(decoded_m_imgs, (decoded_m_imgs.shape[0],40*40))
x_test_flat = np.reshape(x_test, (x_test.shape[0],40*40))
m_images_flat = np.reshape(m_images, (m_images.shape[0],40*40))

mse_bkg = np.mean(np.power(x_test_flat-decoded_bkg_imgs_flat, 2), axis=1)
mse_m = np.mean(np.power(m_images_flat-decoded_m_imgs_flat, 2), axis=1)

plt.hist(mse_bkg,label="bkg",alpha=0.5,bins=50)
plt.hist(mse_m,label="m",alpha=0.5,bins=50)
plt.legend()
plt.yscale("log")
plt.savefig("mse.png")

cm = [[0,0],
      [0,0]]
for elm in mse_bkg:
    if(elm > 0.0045):
        cm[0][1]+=1
    else:
        cm[0][0]+=1
for elm in mse_m:
    if(elm > 0.004):
        cm[1][1]+=1
    else:
        cm[1][0]+=1
print(cm)

