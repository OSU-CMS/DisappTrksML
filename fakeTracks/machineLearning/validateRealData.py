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
import utilities
from fakeClass import fakeNN

def buildModel(filters = [16, 8], input_dim = 55, batch_norm = False, metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]):
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


if __name__ == "__main__":


    ################config parameters################
    weightsDir = '/data/users/mcarrigan/fakeTracks/networks/input_search/inputRemoval/fakeTracks_4PlusLayer_aMCv9p1_9_29_NGBoost_layerEncoded/fakeTracks_4PlusLayer_aMCv9p1_9_29_NGBoost_layerEncoded_p8/weights/'
    plotsName = 'validation_singleMuon2017F'
    workDir = '/data/users/mcarrigan/fakeTracks/'
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_ZToMuMu_v9p1/"]
    val_metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    batch_norm = True
    batch_size = 64
    epochs = 1
    filters = [16, 8]
    input_dim = 178
    undersample = -1
    #delete_elements = []
    delete_elements = ['eventNumber', 'layer1', 'subDet1', 'stripSelection1', 'hitPosX1', 'hitPosY1','layer2', 'subDet2', 'stripSelection2', 'hitPosX2', 'hitPosY2',
                       'layer3', 'subDet3', 'stripSelection3', 'hitPosX3', 'hitPosY3', 'layer4', 'subDet4', 'stripSelection4', 'hitPosX4', 'hitPosY4',
                       'layer5', 'subDet5', 'stripSelection5', 'hitPosX5', 'hitPosY5', 'layer6', 'subDet6', 'stripSelection6', 'hitPosX6', 'hitPosY6',
                       'layer7', 'subDet7', 'stripSelection7', 'hitPosX7', 'hitPosY7', 'layer8', 'subDet8', 'stripSelection8', 'hitPosX8', 'hitPosY8',
                       'layer9', 'subDet9', 'stripSelection9', 'hitPosX9', 'hitPosY9', 'layer10', 'subDet10', 'stripSelection10', 'hitPosX10', 'hitPosY10',
                       'layer11', 'subDet11', 'stripSelection11', 'hitPosX11', 'hitPosY11', 'layer12', 'subDet12', 'stripSelection12', 'hitPosX12', 'hitPosY12',
                       'layer13', 'subDet13', 'stripSelection13', 'hitPosX13', 'hitPosY13', 'layer14', 'subDet14', 'stripSelection14', 'hitPosX14', 'hitPosY14',
                       'layer15', 'subDet15', 'stripSelection15', 'hitPosX15', 'hitPosY15', 'layer16', 'subDet16', 'stripSelection16', 'hitPosX16', 'hitPosY16']
    saveCategories = [{'fake':False, 'real':True, 'pileup':False}]
    normalize_data = False
    DEBUG = False
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

    if 'eventNumber' not in delete_elements:
        delete_elements.append('eventNumber')

    inputs, input_dim = utilities.getInputs(input_dim, delete_elements)
    
    for i, dataSet in enumerate(dataDir):
        if i == 0:
            tracks = utilities.loadData(str(dataSet), undersample, inputs, normalize_data, saveCategories[i], 0, 0, DEBUG)
        else:
            tracks2 = utilities.loadData(str(dataSet), undersample, inputs, normalize_data, saveCategories[i], 0, 0, DEBUG)
            tracks = np.concatenate((tracks, tracks2))


    tracks = tracks[0]

    indices = np.arange(len(tracks))
    np.random.shuffle(indices)   
    tracks = tracks[indices]

    print("tracks shape", tracks.shape)

    model = fakeNN(filters, input_dim, batch_norm, val_metrics)
    estimator = KerasClassifier(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=1)

    #dummy fit to get estimator compiled
    history = estimator.fit(tracks, np.zeros(len(tracks)))
    
    estimator.model.load_weights(weightsDir + 'lastEpoch.h5')
    predictions = estimator.predict_proba(tracks)
    predictions = predictions[:, 1]

    pred_fakes = np.argwhere(predictions >= 0.5)
    pred_reals = np.argwhere(predictions < 0.5)
    
    plotMetrics.plotScores(predictions, np.zeros(len(predictions)), plotsName, plotDir)

    print("Number of predicted fakes: " + str(len(pred_fakes)) + ", Number of predicted Reals: " + str(len(pred_reals)))
           #    h_recall.SetBinContent(iBin, recall)
    
    inputs = utilities.listVariables(inputs)
    
    d0Index = np.where(inputs == 'd0')
    plotMetrics.backgroundEstimation(tracks, predictions, d0Index, plotDir)

    np.savez_compressed(outputDir + "predictions.npz", tracks = tracks, predictions = predictions, inputs = inputs)



