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
import utils

#set directories
workDir = "/home/mcarrigan/disTracksML/autoencoder/"
saveDir = "/data/disappearingTracks/cleaned/N0p7/"
dataDir = "/data/disappearingTracks/electron_selection_DYJetsToll_M50/"

weights = "/home/mcarrigan/disTracksML/weights/autoencoder/eSelect/W_0p25_tanh_N0p7.h5"

trainE = ["_13", "_14", "_15", "_16"]
trainB = ["1001", "1002", "1003", "1004", "1005"]

tag = '_Clean_N0p7'

pix_cut = 30

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
print("ok")


def reshapeData(data):
    images = []
    for i in range(len(data)):
        matrix = data[i, :, :, :]
        matrix = matrix.flatten().reshape([matrix.shape[0]*matrix.shape[1]*matrix.shape[2],])  
        matrix = matrix.astype('float32')
        matrix = np.append(i,matrix)
        images.append(matrix)
    return images

def clean(data):
    for i in range(1):
        x_test = data
        if(x_test.shape[0] == 0): continue
        x_ECAL = x_test[:, :, :, 0]
        x_ECAL = np.reshape(x_ECAL, (-1, 40, 40, 1))
        shouldClean = []
        for j in range(len(x_ECAL)):
            pix_count = 0
            pixels = x_ECAL[j].flatten()
            for pix in range(len(pixels)): 
                if pixels[pix] > 0: pix_count += 1
            if pix_count > pix_cut: shouldClean.append(1)
            #if np.linalg.norm(x_ECAL[j]) >= 1: shouldClean.append(1)
            else: shouldClean.append(0)
        denoised = sortClean(shouldClean, x_ECAL)
        hcal = np.reshape(x_test[:, :, :, 1], (-1, 40, 40, 1))
        muons = np.reshape(x_test[:, :, :, 2], (-1, 40, 40, 1))
        cleaned = np.concatenate((denoised, hcal, muons), axis=3)
        print("denoised shape", denoised.shape)
        print("clean shape", cleaned.shape)
        data = cleaned
    return data

def sortClean(shouldClean, x_ECAL):
    for j in range(len(x_ECAL)):
        if j == 0:
            this_img = np.reshape(x_ECAL[0], (1, 40, 40, 1))
            if shouldClean[0]==1: this_img = autoencoder.predict(this_img)
            denoised = this_img
        else:
            this_img = np.reshape(x_ECAL[j], (1, 40, 40, 1))
            if shouldClean[j]==1: this_img = autoencoder.predict(this_img)
            denoised = np.concatenate((denoised, this_img))
    return denoised

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
    indicies = np.arange(data1.shape[0])
    np.random.shuffle(indicies)
    data1 = data1[indicies]
    data2 = data2[indicies]
    classes = classes[indicies]
    return data1, data2, classes


for file in os.listdir(directory):
    skip = False
    filename = str(file)
    if "bkg" in filename: 
        for x in trainB:
            if x in filename: skip = True
    if "e_" in filename:
        for x in trainE:
            if x in filename: skip = True
    if skip == True: continue
    if filename.endswith(".npz"):
        print("Working on file: " + filename)
        imgName = filename.split(".")
        imgName = imgName[0]
        images, info = utils.load_electron_data(dataDir, filename)
        if len(images) == 0: continue
        data = images[:,1:]
        data = np.reshape(data, [len(data),40,40,4])
        data = data[:,:,:,[0,2,3]]
        x_clean = clean(data)
        data_clean = reshapeData(x_clean)
        data_orig = reshapeData(data) 
        print("clean shape", x_clean.shape)
        np.savez_compressed(saveDir + imgName + tag, images=data_clean, infos = info)
        print("Files saved")
        


