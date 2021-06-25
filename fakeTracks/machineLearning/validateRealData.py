import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import json
import random
import sys
import pickle
import datetime
import getopt
import plotMetrics
from datetime import date

def buildModel(filters = [12, 8], input_dim = 64, batch_norm = True, metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]):
    #begin NN model
    model = Sequential()
    model.add(Dense(filters[0], input_dim=input_dim, activation='relu'))
    for i in range(len(filters)-1):
        model.add(Dense(filters[i+1], activation='relu'))
        if(batch_norm): model.add(BatchNormalization())
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    print(model.summary())
    return model

def callModel():
    model = buildModel(filters, input_dim, batch_norm)
    return model

def loadData(dataDir):
    #layerId, charge, isPixel, pixelHitSize, pixelHitSizeX, pixelHitSizeY, stripShapeSelection, hitPosX, hitPosY
    # load the dataset
    file_count = 0
    for filename in os.listdir(dataDir):
        if('events' and '.npz' not in filename): continue
        print("Loading...", dataDir + filename)
        #if file_count > 10: break
        myfile = np.load(dataDir+filename)
        reals = np.array(myfile["real_infos"])
        if(len(reals)==0): continue
        print(reals.shape, reals.shape[1])
        if(reals.shape[1] < 171): reals = np.hstack((reals, np.zeros((len(reals), 163 - reals.shape[1]))))
        if(file_count == 0):
            realTracks = reals
        else:
            print(realTracks.shape, reals.shape)
            realTracks = np.concatenate((realTracks, reals))
        file_count += 1


    print("Number of tracks:", len(realTracks))

    return realTracks



if __name__ == "__main__":


    ################config parameters################
    weightsDir = '/data/users/mcarrigan/fakeTracks_4PlusLayer_PUveto0p1_aMCv8p1_4_14/fakeTracks_4PlusLayer_PUveto0p1_aMCv8p1_4_14_p3/weights/'
    workDir = '/data/users/mcarrigan/fakeTracks/'
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_higgsino_700_10000_4PlusLayer_v9/"]
    metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    batch_norm = True
    filters = [24, 12]
    input_dim = 171
    #################################################

    if(len(sys.argv) > 1):
        workDir = workDir + sys.argv[1] 
    if(len(sys.argv) > 2):
        dataDir = [sys.argv[2]]

    plotDir = workDir + '/plots/'
    outputDir = workDir + '/outputFiles/'
    
    # create output directories
    os.system('mkdir '+str(workDir))
    os.system('mkdir '+str(plotDir))
    os.system('mkdir '+str(outputDir))
    
    for i, dataSet in enumerate(dataDir):
        if i == 0:
            tracks = loadData(str(dataSet))
        else:
            tracks2 = loadData(str(dataSet))
            tracks = np.concatenate((tracks, tracks2))

    print("Total Tracks: " + str(tracks.shape))

    indices = np.arange(len(tracks))
    np.random.shuffle(indices)   
    tracks = tracks[indices]

    model = callModel()

    model.load_weights(weightsDir + 'lastEpoch.h5')

    predictions = model.predict(tracks)
    
    pred_fakes = np.argwhere(predictions >= 0.5)
    pred_reals = np.argwhere(predictions < 0.5)

    print("Number of predicted fakes: " + str(len(pred_fakes)) + ", Number of predicted Reals: " + str(len(pred_reals)))

    #plotMetrics.predictionCorrelation(predictions, d0, 20, -1, 1, 'predictionD0Correlation', plotDir)
    #plotMetrics.comparePredictions(predictions, d0, 20, -1, 1, 'd0', plotDir)
    plotMetrics.plotScores(predictions, sys.argv[1], plotDir)
    np.savez_compressed(outputDir + "predictions.npz", tracks = tracks, predictions = predictions)



