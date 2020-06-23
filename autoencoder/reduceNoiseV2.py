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
saveDir = "/data/disappearingTracks/cleaned/"
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
autoencoder.load_weights(weights)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#autoencoder.fit(noisyTrain, x_train, epochs = num_epochs, batch_size = batches, shuffle=True, validation_data = (noisyVal, x_val))
print("ok")


def reshapeData(df):
    df_e = df.loc[df['type'] == 1]
    df_bkg = df.loc[df['type'] ==0]
    df_m = df.loc[df['type'] == 2]
    data_e = df_e.iloc[:, 4:].to_numpy()
    data_e = np.reshape(data_e, (data_e.shape[0], 40, 40, 4))
    data_e = data_e[:, :, :, [0, 2, 3]]
    data_bkg = df_bkg.iloc[:, 4:].to_numpy()
    data_bkg = np.reshape(data_bkg, (data_bkg.shape[0],40,40,4))
    data_bkg = data_bkg[:, :, :, [0,2,3]]
    data_m = df_m.iloc[:, 4:].to_numpy()
    data_m = np.reshape(data_m, (-1, 40, 40, 4))
    data_m = data_m[:, :, :, [0,2,3]]
    data = [data_bkg, data_e, data_m]
    fig, plot = plt.subplots(1,5, figsize = (10,5))
    #for j in range(5):
        #rand = np.random.randint(0, data[1].shape[0])
        #plot[j].imshow(data[1][rand, :, :, 0])
        #plt.savefig(workDir + 'ReshapedImg.png')

    return data


def clean(data):
    for i in range(3):
        x_test = data[i]
        #print(x_test.shape[0])
        if(x_test.shape[0] == 0): continue
        x_ECAL = x_test[:, :, :, 0]
        #reshape data to be 4 dimensions for nn
        x_ECAL = np.reshape(x_ECAL, (-1, 40, 40, 1))
        denoised = autoencoder.predict(x_ECAL)
        #denoised = np.reshape(denoised, (-1, 40, 40))
        hcal = np.reshape(x_test[:, :, :, 1], (-1, 40, 40, 1))
        muons = np.reshape(x_test[:, :, :, 2], (-1, 40, 40, 1))
        cleaned = np.concatenate((denoised, hcal, muons), axis=3)
        #cleaned = np.reshape(cleaned, (-1, 40, 40, 3))
        print("denoised shape", denoised.shape)
        print("clean shape", cleaned.shape)
        data[i] = cleaned
        #print("data[i] shape", data[i].shape)
        #if i == 1:
            #fig, plot = plt.subplots(3,5, figsize = (8,5))
            #for j in range(5):
                #rand = np.random.randint(0, cleaned.shape[0])
                #plot[0,j].imshow(denoised[rand, :, :,0])
                #plot[1,j].imshow(cleaned[rand, :, :, 0])
                #plot[2,j].imshow(x_ECAL[rand, :, :, 0])
            #plt.savefig(workDir + 'CleanedImg.png')
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
   

def shuffleData(data1, data2, classes):
    #shuffle data
    #print(data1.shape)
    #print(data2.shape)
    indicies = np.arange(data1.shape[0])
    np.random.shuffle(indicies)
    data1 = data1[indicies]
    data2 = data2[indicies]
    classes = classes[indicies]
    return data1, data2, classes


for file in os.listdir(directory):
    #filename = os.fsdecode(file)
    filename = str(file)
    if filename.endswith(".pkl"):
        print("Working on file: " + filename)
        saveName = filename.split(".")
        imgName = saveName[0]
        className = imgName.split("_")
        className = className[2] 
        #print(className)
        df = pd.read_pickle(dataDir + filename)
        x_orig = reshapeData(df)
        x_clean = clean(x_orig)
        x_orig = reshapeData(df)
        x_orig = np.concatenate((x_orig[0], x_orig[1], x_orig[2]))
        x_clean = np.concatenate((x_clean[0], x_clean[1], x_clean[2]))
        fig, plot = plt.subplots(2,5, figsize = (8,5))
        #for j in range(5):
            #rand = np.random.randint(0, x_clean.shape[0])
            #plot[0,j].imshow(x_orig[rand, :, :, 0])
            #plot[1,j].imshow(x_clean[rand, :, :, 0])
        #plt.savefig(workDir + 'TestImg.png')
        print("clean shape", x_clean.shape)
        print("orig shape", x_orig.shape)
        att = getAttributes(df)
        data_clean, data_orig, classes = shuffleData(x_clean, x_orig,  att)
        np.save(saveDir + imgName + '_Clean.npy', data_clean)
        #np.save(saveDir + imgName + '_Orig.npy', data_orig)
        np.save(saveDir + 'classes_0p25' + className + '_Clean.npy', classes)
        print("Files saved")
        


