import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import utils
import json
import random
import sys
import cnn

dataDir = "/data/disappearingTracks/"
workDir = '/home/llavezzo/'
weightsDir = workDir + 'weights/cnn/'
fname = "images_0p25_tanh_singleElectron2017.npz"
weightsFile = 'first_model'


################config parameters################
img_rows, img_cols = 40, 40
channels = 3
input_shape = (img_rows,img_cols,channels)   
#################################################

data = np.load(dataDir+fname)
images = data['images']
images = images[:,1:]
images = np.reshape(images,(images.shape[0],40,40,4))
x_test = images[:,:,:,[0,2,3]]

model = cnn.build_model(input_shape = input_shape, 
                    layers = 5, filters = 64, opt='adam')

model.load_weights(weightsDir+weightsFile+'.h5')

predictions = model.predict(x_test)

isElectron,isBkg = 0,0
for pred in predictions:
    if pred[1] > 0.1: isElectron+=1
    else: isBkg+=1
print(isElectron,isBkg)