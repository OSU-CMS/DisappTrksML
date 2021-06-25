import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import json
import random
import pickle
import datetime
import getopt
import plotMetrics
from datetime import date
from sklearn.preprocessing import MinMaxScaler


variables = ['passesSelection', 'nPV', 'trackIso', 'eta', 'phi', 'nValidPixelHits', 'nValidHits', 'missingOuterHits', 'dEdxPixel', 'dEdxStrip', 'numMeasurementsPixel', 'numMeasurementsStrip', 'numSatMeasurementsPixel', 'numSatMeasurementsStrip', 'dRMinJet', 'ecalo', 'pt', 'd0', 'dz', 'totalCharge', 'deltaRToClosestElectron', 

    'layer1', 'charge1', 'subDet1', 'pixelHitSize1', 'pixelHitSizeX1', 
           'pixelHitSizeY1','stripSelection1', 'hitPosX1', 'hitPosY1', 
    'layer2', 'charge2', 'subDet2', 'pixelHitSize2', 'pixelHitSizeX2', 
           'pixelHitSizeY2', 'stripSelection2', 'hitPosX2', 'hitPosY2', 
    'layer3', 'charge3', 'subDet3', 'pixelHitSize3', 'pixelHitSizeX3', 
           'pixelHitSizeY3', 'stripSelection3', 'hitPosX3', 'hitPosY3', 
    'layer4', 'charge4', 'subDet4', 'pixelHitSize4', 'pixelHitSizeX4', 
           'pixelHitSizeY4', 'stripSelection4', 'hitPosX4', 'hitPosY4', 
    'layer5', 'charge5', 'subDet5', 'pixelHitSize5', 'pixelHitSizeX5', 
           'pixelHitSizeY5', 'stripSelection5', 'hitPosX5', 'hitPosY5', 
    'layer6', 'charge6', 'subDet6', 'pixelHitSize6', 'pixelHitSizeX6', 
           'pixelHitSizeY6', 'stripSelection6', 'hitPosX6', 'hitPosY6',
    'layer7', 'charge7', 'subDet7', 'pixelHitSize7', 'pixelHitSizeX7', 
           'pixelHitSizeY7', 'stripSelection7', 'hitPosX7', 'hitPosY7',
    'layer8', 'charge8', 'subDet8', 'pixelHitSize8', 'pixelHitSizeX8', 
           'pixelHitSizeY8', 'stripSelection8', 'hitPosX8', 'hitPosY8',
    'layer9', 'charge9', 'subDet9', 'pixelHitSize9', 'pixelHitSizeX9', 
           'pixelHitSizeY9', 'stripSelection9', 'hitPosX9', 'hitPosY9',
    'layer10', 'charge10', 'subDet10', 'pixelHitSize10', 'pixelHitSizeX10', 
           'pixelHitSizeY10', 'stripSelection10', 'hitPosX10', 'hitPosY10',
    'layer11', 'charge11', 'subDet11', 'pixelHitSize11', 'pixelHitSizeX11', 
           'pixelHitSizeY11', 'stripSelection11', 'hitPosX11', 'hitPosY11', 
    'layer12', 'charge12', 'subDet12', 'pixelHitSize12', 'pixelHitSizeX12', 
           'pixelHitSizeY12', 'stripSelection12', 'hitPosX12', 'hitPosY12',
    'layer13', 'charge13', 'subDet13', 'pixelHitSize13', 'pixelHitSizeX13', 
           'pixelHitSizeY13', 'stripSelection13', 'hitPosX13', 'hitPosY13',
    'layer14', 'charge14', 'subDet14', 'pixelHitSize14', 'pixelHitSizeX14', 
           'pixelHitSizeY14', 'stripSelection14', 'hitPosX14', 'hitPosY14',
    'layer15', 'charge15', 'subDet15', 'pixelHitSize15', 'pixelHitSizeX15', 
           'pixelHitSizeY15', 'stripSelection15', 'hitPosX15', 'hitPosY15',
    'layer16', 'charge16', 'subDet16', 'pixelHitSize16', 'pixelHitSizeX16', 
           'pixelHitSizeY16', 'stripSelection16', 'hitPosX16', 'hitPosY16',

    'sumEnergy', 'diffEnergy', 'dz1', 'd01', 'dz2', 'd02', 'dz3', 'd03']

varDict = {}
for x in range(len(variables)):
    varDict[variables[x]] = x

def loadData(dataDir, undersample, inputs, normalize_data):
    #layerId, charge, isPixel, pixelHitSize, pixelHitSizeX, pixelHitSizeY, stripShapeSelection, hitPosX, hitPosY
    # load the dataset
    file_count = 0
    realTracks = []
    fakeTracks = []
    for filename in os.listdir(dataDir):
        print("Loading...", dataDir + filename)
        #if file_count > 20: break
        myfile = np.load(dataDir+filename)
        fakes = np.array(myfile["fake_infos"])
        fakes = selectInputs(fakes, inputs)
        reals = np.array(myfile["real_infos"])
        reals = selectInputs(reals, inputs)
        if(file_count == 0):
            fakeTracks = fakes
            realTracks = reals
        elif(file_count != 0 and len(fakeTracks) == 0): fakeTracks = fakes
        elif(file_count != 0 and len(realTracks) == 0): realTracks = reals
        else:
            if(len(fakes)!=0): fakeTracks = np.concatenate((fakeTracks, fakes))
            if(len(reals)!=0): realTracks = np.concatenate((realTracks, reals))
        file_count += 1


    print("Number of fake tracks:", len(fakeTracks))
    print("Number of real tracks:", len(realTracks))

    trainRealTracks, testRealTracks, trainRealTruth, testRealTruth = train_test_split(realTracks, np.zeros(len(realTracks)), test_size = 0.3)
    trainFakeTracks, testFakeTracks, trainFakeTruth, testFakeTruth = train_test_split(fakeTracks, np.ones(len(fakeTracks)), test_size = 0.3)

    testRealTracks, valRealTracks, testRealTruth, valRealTruth = train_test_split(testRealTracks, testRealTruth, test_size = 0.5)
    testFakeTracks, valFakeTracks, testFakeTruth, valFakeTruth = train_test_split(testFakeTracks, testFakeTruth, test_size = 0.5)

    # if undersampling
    if(undersample != -1):
        num_real = len(trainRealTracks)
        num_select = int(undersample * num_real)
        ind = np.arange(num_real)
        ind = np.random.choice(ind, num_select)
        trainRealTracks = trainRealTracks[ind]


    #combine all data and shuffle
    trainTracks = np.concatenate((trainFakeTracks, trainRealTracks))
    testTracks = np.concatenate((testFakeTracks, testRealTracks))
    trainTruth = np.concatenate((trainFakeTruth, trainRealTruth))
    testTruth = np.concatenate((testFakeTruth, testRealTruth))
    valTracks = np.concatenate((valFakeTracks, valRealTracks))
    valTruth = np.concatenate((valFakeTruth, valRealTruth))

    # Apply min max scale over all data (scale set range [-1,1])
    #scaler = MinMaxScaler(feature_range=(-1,1), copy=False)
    #scaler.partial_fit(trainTracks)
    #scaler.partial_fit(testTracks)
    #scaler.partial_fit(valTracks)
    #testTracks = scaler.transform(testTracks)
    #trainTracks = scaler.transform(trainTracks)
    #valTracks = scaler.transform(valTracks)

    test_indices = np.arange(len(testTracks))
    np.random.shuffle(test_indices)
    train_indices = np.arange(len(trainTracks))
    np.random.shuffle(train_indices)
    val_indices = np.arange(len(valTracks))
    np.random.shuffle(val_indices)

    trainTracks = trainTracks[train_indices]
    trainTracks = np.reshape(trainTracks, (-1,len(inputs)))
    if(normalize_data): trainTracks = np.tanh(trainTracks)
    trainTruth = trainTruth[train_indices]

    testTracks = testTracks[test_indices]
    testTracks = np.reshape(testTracks, (-1,len(inputs)))
    if(normalize_data): testTracks = np.tanh(testTracks)
    testTruth = testTruth[test_indices]

    valTracks = valTracks[val_indices]
    valTracks = np.reshape(valTracks, (-1,len(inputs)))
    if(normalize_data): valTracks = np.tanh(valTracks)
    valTruth = valTruth[val_indices]

    return trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth

def selectInputs(array, inputs):
    if inputs.any() == -1: return array

    array = array[:, inputs]
    return array

def getInputs(input_dim, delete):
    
    if len(varDict) == 0: createDict(variables)

    inputs = np.arange(0, input_dim)

    delete_array = []
    for x in varDict:
        for y in delete:
            if y in x:
                print(x, varDict[x])
                delete_array.append(varDict[x])

    inputs = np.delete(inputs, delete_array)
    input_dim = len(inputs)
    print(input_dim)
    return inputs, input_dim

def createDict(variables):
    varDict = {}
    for x in range(len(variables)):
        varDict[variables[x]] = x

    return varDict



if __name__ == '__main__':

    for x in range(len(variables)):
        varDict[variables[x]] = x

    print(varDict)

    #inputs = np.arange(0, len(variables))


    delete_elements = ['totalCharge', 'numSatMeasurements', 'stripSelection', 'hitPosX', 'hitPosY']
    #delete_array = []
    #for x in varDict:
    #    for y in delete_elements:
    #        if y in x:
    #            print(x, varDict[x]) 
    #            delete_array.append(varDict[x])

    #inputs = np.delete(inputs, delete_array)
    #inputs = np.delete(inputs, varDict['charge'])
    #print(inputs, len(inputs))

    undersample = -1
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_v9_DYJets_aMCNLO_4PlusLayer_v9p1/"]
    input_dim = len(variables)
    normalize_data = False


    inputs, input_dim = getInputs(input_dim, delete_elements)
    trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = loadData(str(dataDir[0]), undersample)
    print("shape of array", trainTracks.shape)











