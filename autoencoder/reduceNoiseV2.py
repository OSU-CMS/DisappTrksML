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
import os

#set epochs, batch size
num_epochs = 30
batches = 16

#set directories
workDir = "/home/mcarrigan/disTracksML/autoencoder/"
saveDir = "/home/mcarrigan/disTracksML/data/v2/cleaned/"
dataDir = "/home/mcarrigan/disTracksML/data/v2/"

weights = "/home/mcarrigan/disTracksML/weights/autoencoder/W_0p25_tanh.h5"

#directory = os.fsencode(dataDir)
directory = dataDir

img = Input(shape = (40,40,1))
ae = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(img)
encoded = MaxPooling2D((2,2), padding = 'same')(ae)
ae = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(encoded)
ae = UpSampling2D((2,2))(ae)
decoded = Conv2D(1, (3,3), activation = 'sigmoid', padding = 'same')(ae)
autoencoder = Model(img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.fit(noisyTrain, x_train, epochs = num_epochs, batch_size = batches, shuffle=True, validation_data = (noisyVal, x_val))
autoencoder.load_weights(weights)
print("ok")


def reshapeData(df):
    df_e = df.loc[df['type'] == 1]
    df_bkg = df.loc[df['type'] ==0]
    df_m = df.loc[df['type'] == 2]
    data_e = df_e.iloc[:, 4:].to_numpy()
    data_e = np.reshape(data_e, (data_e.shape[0], 40, 40, 4))
    data_e = data_e[:, :, :, :]
    data_bkg = df_bkg.iloc[:, 4:].to_numpy()
    data_bkg = np.reshape(data_bkg, (data_bkg.shape[0],40,40,4))
    data_bkg = data_bkg[:, :, :, :]
    data_m = df_m.iloc[:, 4:].to_numpy()
    data_m = np.reshape(data_m, (-1, 40, 40, 4))
    data_m = data_m[:, :, :, :]
    data = [data_bkg, data_e, data_m]
    for i in range(len(data)):
        x_test = data[i]
        #print(x_test.shape[0])
        if(x_test.shape[0] == 0): continue
        x_ECAL = x_test[:, :, :, 0]
        #reshape data to be 4 dimensions for nn
        x_ECAL = np.reshape(x_ECAL, (-1, 40, 40, 1))
        denoised = autoencoder.predict(x_ECAL)
        denoised = np.reshape(denoised, (-1, 40, 40))
        denoised = np.array([denoised, x_test[:, :, :, 1], x_test[:, :, :, 2], x_test[:, :, :, 3]])
        denoised = np.reshape(denoised, (-1, 40, 40, 4))
        #print(denoised.shape)
        data[i] = denoised
    return data


def getAttributes(df):
    df_e = df.loc[df['type'] == 1]
    df_bkg = df.loc[df['type'] ==0]
    df_m = df.loc[df['type'] == 2]
    att_e = df_e.iloc[:, :4].to_numpy()
    att_bkg = df_bkg.iloc[:, :4].to_numpy()
    att_m = df_m.iloc[:, :4].to_numpy()
    att = np.concatenate((att_bkg, att_e, att_m))
    return att
   

def shuffleData(data, classes):
    #shuffle data
    indicies = np.arange(data.shape[0])
    np.random.shuffle(indicies)
    data = data[indicies]
    classes = classes[indicies]
    return data, classes


for file in os.listdir(directory):
    #filename = os.fsdecode(file)
    filename = str(file)
    if filename.endswith(".pkl"):
        print("Working on file: " + filename)
        saveName = filename.split(".")
        imgName = saveName[0]
        className = imgName.split("_")
        className = className[1] 
        df = pd.read_pickle(dataDir + filename)
        x_test = reshapeData(df)
        x_test = np.concatenate((x_test[0], x_test[1], x_test[2]))
        att = getAttributes(df)
        data, classes = shuffleData(x_test, att)
        np.save(saveDir + imgName + '_Clean.npy', data)
        np.save(saveDir + 'classes' + className + '_Clean.npy', classes)
        
        


