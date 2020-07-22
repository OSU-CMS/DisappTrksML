import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras import optimizers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import scipy
import keras
import sys
import utils
import os

if(len(sys.argv) > 2): loadWeights = str(sys.argv[2])
saveWeights = str(sys.argv[1])


#set epochs, batch size
num_epochs = 10
batches = 16

pct_noise = 0.7

#set directories
workDir = "/home/mcarrigan/disTracksML/autoencoder/"
saveDir = "/home/mcarrigan/disTracksML/weights/autoencoder/eSelect/"
dataDir = "/data/disappearingTracks/AE/"
#dataDir = "/home/mcarrigan/disTracksML/data/v2/AE/"


def reshapeUp(data):
    data = np.reshape(data, (-1, 40, 40, 3))
    return data

def removeEle(data, index):
    data1 = data[:index]
    data2 = data[index+1:]
    data = np.concatenate((data1, data2))
    return data

e_count = 0
bkg_count = 0
file_count = 0
#noise_nPV = []

def reshapeUp(data):
    data = np.reshape(data, (-1, 40, 40, 3))
    return data

for filename in os.listdir(dataDir):
    if "e_" not in filename: continue
    images, info = utils.load_electron_data(dataDir, filename)
    if len(images) ==0: continue
    data = images[:,1:]
    data = np.reshape(data, [len(data),40,40,4])
    data = data[:,:,:,[0,2,3]]
    classes = np.ones(len(data))
    #nPV = nPV.astype(int)
    for i in range(len(classes)):
        if(i%100 == 0): print("Working on electron event ", i)
        #if classes[i] == 1: print("electron")
        if np.linalg.norm(data[i, :, :, 0]) == 0: continue
        if e_count == 0: data_e = reshapeUp(data[i, :, :, :]) 
        else: data_e = np.concatenate((data_e, reshapeUp(data[i, :, :, :])))
        e_count += 1

print("Total Electron Events: ", len(data_e))        

for filename in os.listdir(dataDir):
    if "bkg" not in filename: continue
    bkg_cout = 0
    images, info = utils.load_electron_data(dataDir, filename)
    if len(images) == 0: continue
    data = images[:,1:]
    data = np.reshape(data, [len(data),40,40,4])
    data = data[:,:,:,[0,2,3]]
    classes = np.array([x[1] for x in info])
    #nPV = [x[2] for x in info]
    classes = classes.astype(int)
    #nPV = nPV.astype(int)
    for i in range(len(classes)):
        if(i%100 == 0): print("Working on background event ", i)
        if classes[i] != 1: 
            if np.linalg.norm(data[i, :, :, 0]) == 0: continue
            if bkg_count == 0: this_bkg = reshapeUp(data[i, :, :, :])
            else:this_bkg = np.concatenate((this_bkg, reshapeUp(data[i, :, :, :])))
            bkg_count += 1
    if file_count == 0: data_bkg = this_bkg
    else: data_bkg = np.concatenate((data_bkg, this_bkg))
    file_count += 1
    
    
#shuffle data
indicies = np.arange(data_e.shape[0])
np.random.shuffle(indicies)
data_e = data_e[indicies]
classes_e = np.ones(len(data_e))
print("Background events ", data_bkg.shape)
x_train, x_val, y_train, y_val = train_test_split(data_e, classes_e, test_size=0.15, random_state=0)

print("Training Samples", x_train.shape[0])
print("Validation Samples", x_val.shape[0])

x_train = x_train[:, :, :, 0]
x_val = x_val[:, :, :, 0]

fig, plot = plt.subplots(1,5, figsize = (10,5))
for i in range(5):
    rand = np.random.randint(0, x_train.shape[0])
    plot[i].imshow(x_train[rand, :, :])

plt.savefig(workDir + 'TrainElectrons.png')

x_train = np.reshape(x_train, (-1, 40, 40, 1))
x_val = np.reshape(x_val, (-1, 40, 40, 1))

print("Train shape", x_train.shape)
print("Val shape", x_val.shape)

def noisyData(data_e, data_bkg, pct_noise):
    evts = len(data_e)
    bkg_used = np.zeros(len(data_bkg))
    #noise = np.random.randint(0, len(data_bkg), evts)
    #data_bkg = data_bkg[noise]
    data_noise = np.copy(data_e[:, :, :, 0])
    data_noise = np.reshape(data_noise, (-1, 40, 40, 1))
    for i in range(len(data_e)):
        if i% 500 ==0: print("Adding noise to electron event...", i)
        evt_added = 0
        norm_bkg = 0
        normE = np.linalg.norm(data_e[i, :, :, 0])
        while (norm_bkg < normE * pct_noise):
            index = np.random.randint(0, len(data_bkg))
            if(bkg_used[index] == 1): continue
            if(evt_added ==0): this_bkg = data_bkg[index, :, :, 0]
            else: this_bkg = np.add(this_bkg, data_bkg[index, :, :, 0])
            bkg_used[index] = 1
            norm_bkg = np.linalg.norm(this_bkg)
            evt_added += 1
        this_bkg = np.reshape(this_bkg, (1, 40, 40, 1))
        #print(this_bkg.shape)
        data_noise[i, :, :, 0] = np.add(data_noise[i, :, :, 0], this_bkg[0, :, :, 0])
        data_noise = np.reshape(data_noise, (-1, 40, 40, 1))
    return data_noise

print("Adding noise...")
noisyTrain = noisyData(x_train, data_bkg, pct_noise)
noisyVal = noisyData(x_val, data_bkg, pct_noise)
print("Noise added, training autoencoder...")


img = Input(shape = (40,40,1))
ae = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(img)
#ae = MaxPooling2D((2,2), padding = 'same')(ae)
#encoded = Conv2D(32, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(ae)
#ae = Conv2D(32, (3,3), activation = 'relu', padding = "same")(ae)
encoded = MaxPooling2D((2,2), padding = 'same')(ae)
#encoded = Conv2D(16, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(ae)

ae = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(encoded)
#ae = Conv2DTranspose(32, (3,3), activation = 'relu', strides = (2,2), padding = 'same')(ae)
ae = UpSampling2D((2,2))(ae)
#ae = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(ae)
#ae = UpSampling2D((2,2))(ae)
#ae = Conv2DTranspose(128, (3,3), strides = (2,2), activation = 'relu', padding = 'same')(ae)
decoded = Conv2D(1, (3,3), activation = 'sigmoid', padding = 'same')(ae)


autoencoder = Model(img, decoded)
if(len(sys.argv) > 2): 
    autoencoder.load_weights(saveDir + loadWeights)
    print("loading weights", loadWeights)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(noisyTrain, x_train, epochs = num_epochs, batch_size = batches, shuffle=True, validation_data = (noisyVal, x_val))

autoencoder.save_weights(saveDir+saveWeights)

