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

#import utils
#import validate


dataDir = "/store/user/mcarrigan/fakeTracks/converted_v1/"

# load the dataset
fakeTracks = np.array([])
realTracks = np.array([])

for filename in os.listdir(dataDir):
    myfile = np.load(filename)
    fakes = np.array(myfile["fake_infos"])
    reals = np.array(myfile["real_infos"])
    fakeTracks.append(fakes)
    realTracks.append(reals)    

print("Number of fake tracks:", len(fakeTracks))
print("Number of real tracks:", len(realTracks))

# split into input (X) and output (y) variables
#X = dataset[:,0:8]
#y = dataset[:,8]
# define the keras model
#model = Sequential()
#model.add(Dense(12, input_dim=8, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
# compile the keras model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
#model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# make class predictions with the model
#predictions = model.predict_classes(X)
# summarize the first 5 cases
#for i in range(5):
#	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

