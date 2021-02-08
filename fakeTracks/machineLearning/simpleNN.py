import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import random
import sys
import pickle
import datetime
import getopt
import plotMetrics

#import utils
#import validate


dataDir = "/store/user/mcarrigan/fakeTracks/converted_v1/"
weightsDir = "weights/"
metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()] 


#layerId, charge, isPixel, pixelHitSize, pixelHitSizeX, pixelHitSizeY, stripShapeSelection, hitPosX, hitPosY
# load the dataset
file_count = 0
for filename in os.listdir(dataDir):
    print("Loading...", filename)
    if file_count > 10: break
    myfile = np.load(dataDir+filename)
    fakes = np.array(myfile["fake_infos"])
    reals = np.array(myfile["real_infos"])
    if(file_count == 0):
        fakeTracks = fakes
        realTracks = reals
    else:
        fakeTracks = np.concatenate((fakeTracks, fakes))
        realTracks = np.concatenate((realTracks, reals))    
    file_count += 1


print("Number of fake tracks:", len(fakeTracks))
print("Number of real tracks:", len(realTracks))

print("FakeTracks shape", np.shape(fakeTracks))

#print(fakeTracks[0, :, :])

#combine all data and shuffle
allTracks = np.concatenate((fakeTracks, realTracks))

indices = np.arange(len(allTracks))
np.random.shuffle(indices)

allTracks = allTracks[indices]
allTracks = np.reshape(allTracks, (-1,156))
print("allTracks shape", np.shape(allTracks))
print("allTracks", allTracks[0])
allTruth = np.concatenate((np.ones(len(fakeTracks)), np.zeros(len(realTracks))))
allTruth = allTruth[indices]

#split data into train and test

trainTracks, testTracks, trainTruth, testTruth = train_test_split(allTracks, allTruth, test_size = 0.2)

#begin NN model
model = keras.Sequential()
model.add(keras.layers.Dense(12, input_shape=(156,), activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))


model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=metrics)

print(model.summary())

# fit the keras model on the dataset
history = model.fit(trainTracks, trainTruth, epochs=10, batch_size=1000, verbose=1)

# make class predictions with the model
predictions = model.predict_classes(testTracks)

plotMetrics.plotCM(testTruth, predictions)
plotMetrics.getStats(testTruth, predictions)
plotMetrics.plotHistory(history, ['loss','auc', 'recall', 'precision'])


#for i in range(5):
#	print('%d (expected %d)' % (predictions[i], testTruth[i]))
#fakesPredicted = 0
#j = 0
#while fakesPredicted < 5:
#    j += 1
#    if testTruth[j] == 1:
#        fakesPredicted += 1
#        print('%d (expected %d)' % (predictions[j], testTruth[j]))

model.save_weights(weightsDir+'lastEpoch.h5')
np.savez_compressed("predictions_test2.npz", tracks = testTracks, truth = testTruth, predictions = predictions)
