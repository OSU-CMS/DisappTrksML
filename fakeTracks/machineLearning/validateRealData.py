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
from ROOT import TTree, TBranch, TFile
from array import array

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
    #weightsDir = '/data/users/mcarrigan/fakeTracks/networks/input_search/inputRemoval/fakeTracks_4PlusLayer_aMCv9p1_9_29_NGBoost_layerEncoded/fakeTracks_4PlusLayer_aMCv9p1_9_29_NGBoost_layerEncoded_p8/weights/'
    weightsDir = '/data/users/mcarrigan/fakeTracks/networks/dropoutSearch/fakeTracks_4PlusLayer_aMCv9p3_NGBoost_ProducerValidation_3-29_v2/fakeTracks_4PlusLayer_aMCv9p3_NGBoost_ProducerValidation_3-29_v2_p4/weights/'
    plotsName = 'validation_FakeProducer'
    workDir = '/data/users/mcarrigan/fakeTracks/'
    dataDir = ["data/validateProducer/test/"]
    val_metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    batch_norm = False
    batch_size = 64
    epochs = 1
    filters = [16, 8]
    input_dim = 178
    undersample = -1
    dropout = 0.1
    #delete_elements = []
    '''delete_elements = ['eventNumber', 'layer1', 'subDet1', 'stripSelection1', 'hitPosX1', 'hitPosY1','layer2', 'subDet2', 'stripSelection2', 'hitPosX2', 'hitPosY2',
                       'layer3', 'subDet3', 'stripSelection3', 'hitPosX3', 'hitPosY3', 'layer4', 'subDet4', 'stripSelection4', 'hitPosX4', 'hitPosY4',
                       'layer5', 'subDet5', 'stripSelection5', 'hitPosX5', 'hitPosY5', 'layer6', 'subDet6', 'stripSelection6', 'hitPosX6', 'hitPosY6',
                       'layer7', 'subDet7', 'stripSelection7', 'hitPosX7', 'hitPosY7', 'layer8', 'subDet8', 'stripSelection8', 'hitPosX8', 'hitPosY8',
                       'layer9', 'subDet9', 'stripSelection9', 'hitPosX9', 'hitPosY9', 'layer10', 'subDet10', 'stripSelection10', 'hitPosX10', 'hitPosY10',
                       'layer11', 'subDet11', 'stripSelection11', 'hitPosX11', 'hitPosY11', 'layer12', 'subDet12', 'stripSelection12', 'hitPosX12', 'hitPosY12',
                       'layer13', 'subDet13', 'stripSelection13', 'hitPosX13', 'hitPosY13', 'layer14', 'subDet14', 'stripSelection14', 'hitPosX14', 'hitPosY14',
                       'layer15', 'subDet15', 'stripSelection15', 'hitPosX15', 'hitPosY15', 'layer16', 'subDet16', 'stripSelection16', 'hitPosX16', 'hitPosY16']'''
    delete_elements = ['passesSelection']
    saveCategories = [{'fake':True, 'real':True, 'pileup':False}]
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

    #if 'eventNumber' not in delete_elements:
    #    delete_elements.append('eventNumber')

    inputs, input_dim = utilities.getInputs(input_dim, delete_elements)
    
    for i, dataSet in enumerate(dataDir):
        if i == 0:
            #swapped test/train here because we want to validate on "test" data
            testTracks, trainTracks, valTracks, testTruth, trainTruth, valTruth = utilities.loadData(str(dataSet), undersample, inputs, normalize_data, saveCategories[i], 0, 0, DEBUG)
            print("Tracks shape", testTracks.shape)
        else:
            testTracks2, trainTracks2, valTracks2, testTruth2, trainTruth2, valTruth2 = utilities.loadData(str(dataSet), undersample, inputs, normalize_data, saveCategories[i], 0, 0, DEBUG) 
            print("Tracks shape", testTracks2.shape)
            trainTracks = np.concatenate((trainTracks, trainTracks2))
            trainTruth = np.concatenate((trainTruth, trainTruth2))
            testTracks = np.concatenate((testTracks, testTracks2))
            testTruth = np.concatenate((testTruth, testTruth2))
            valTracks = np.concatenate((valTracks, valTracks2))
            valTruth = np.concatenate((valTruth, valTruth2))

    indices = np.arange(len(testTracks))
    np.random.shuffle(indices)   
    testTracks = testTracks[indices]

    eventNumbers = testTracks[:, 0]
    testTracks = testTracks[:, 1:]

    #Code is to run KerasClassifier, easier to save whole model and load NOTE KerasClassifier cannot run predict, due to bug
    #model = fakeNN(filters, input_dim, batch_norm, val_metrics, dropout)
    #model.compile(loss='binary_crossentropy', optimizer='adam')
    #estimator = KerasClassifier(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=1)
    #dummy fit to get estimator compiled
    #history = estimator.fit(testTracks, np.zeros(len(testTracks)))
    
    model = keras.models.load_model(weightsDir + 'lastEpoch.h5')    
    predictions_raw = model.predict_proba(testTracks)
    predictions = model.predict(testTracks)
    plotMetrics.plotCM(testTruth, predictions, plotDir)
    classifications = plotMetrics.getStats(testTruth, predictions, plotDir, plot=True)
    plotMetrics.plotScores(predictions_raw, testTruth, 'fakeNN', plotDir)
    plotMetrics.predictionThreshold(predictions_raw, testTruth, plotDir)

    pred_fakes = np.argwhere(predictions >= 0.5)
    pred_reals = np.argwhere(predictions < 0.5)
    
    plotMetrics.plotScores(predictions, np.zeros(len(predictions)), plotsName, plotDir)

    print("Number of predicted fakes: " + str(len(pred_fakes)) + ", Number of predicted Reals: " + str(len(pred_reals)))
           #    h_recall.SetBinContent(iBin, recall)
    
    inputs = utilities.listVariables(inputs)
    
    d0Index = np.where(inputs == 'd0')
    plotMetrics.backgroundEstimation(testTracks, predictions, d0Index, plotDir)

    np.savez_compressed("predictions.npz", tracks = testTracks, truth = testTruth, predictions = predictions, inputs = inputs, events = eventNumbers)

    r_predictions = array('d', [0.])
    r_truth = array('d', [0.])
    r_event = array('d', [0.])
    r_eta = array('d', [0.])
    r_phi = array('d', [0.])
    r_pt = array('d', [0.])
    r_trackIso = array('d', [0.])
    myfile = TFile("predictions_test.root", "recreate")
    mytree = TTree("predictions", "Network Predictions")
    mytree.Branch("predictions", r_predictions, "r_predictions/D")
    mytree.Branch("truth", r_truth, "r_truth/D")
    mytree.Branch("event", r_event, "r_event/D")
    mytree.Branch("eta", r_eta, "r_eta/D")
    mytree.Branch("phi", r_phi, "r_phi/D")
    mytree.Branch("pt", r_pt, "r_pt/D")
    mytree.Branch("trackIso", r_trackIso, "r_trackIso/D")
    for i in range(len(predictions)):
        r_predictions[0] = predictions[i]
        r_truth[0] = testTruth[i]
        r_event[0] = eventNumbers[i] #event number
        r_trackIso[0] = testTracks[i, 1] #trackIso
        r_eta[0] = testTracks[i, 2] #eta
        r_phi[0] = testTracks[i, 3] #phi
        r_pt[0] = testTracks[i, 15] #pt
        if(r_predictions[0] != predictions[i]): print("problem with prediction in event", eventNumbers[i])
        if(r_truth[0] != testTruth[i]): print("problem with truth in event", eventNumbers[i])
        if(r_event[0] != eventNumbers[i]): print("problem with event number in event", eventNumbers[i])
        if(r_trackIso[0] != testTracks[i, 1]): print("problem with trackIso in event", eventNumbers[i])
        if(r_eta[0] != testTracks[i, 2]): print("problem with eta in event", eventNumbers[i])
        if(r_phi[0] != testTracks[i, 3]): print("problem with phi in event", eventNumbers[i])
        if(r_pt[0] != testTracks[i, 15]): print("problem with pt in event", eventNumbers[i])

        mytree.Fill()

    mytree.Write()
    myfile.Close()







