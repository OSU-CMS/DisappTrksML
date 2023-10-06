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
#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import json
import random
import pickle
import datetime
import getopt
import plotMetrics
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import decimal

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#need to rerun convert and then use this updated variables list
variables = ['eventNumber', 'nPV', 'passesSelection', 'trackIso', 'eta', 'phi', 'nValidPixelHits', 'nValidHits', 'missingOuterHits', 'dEdxPixel', 'dEdxStrip', 'numMeasurementsPixel', 'numMeasurementsStrip', 'numSatMeasurementsPixel', 'numSatMeasurementsStrip', 'dRMinJet', 'ecalo', 'pt', 'd0', 'dz', 'totalCharge', 'deltaRToClosestElectron', 'deltaRToClosestMuon', 'deltaRToClosestTauHad', 'normalizedChi2', 
#variables = ['passesSelection', 'eventNumber', 'nPV', 'trackIso', 'eta', 'phi', 'nValidPixelHits', 'nValidHits', 'missingOuterHits', 'dEdxPixel', 'dEdxStrip', 'numMeasurementsPixel', 'numMeasurementsStrip', 'numSatMeasurementsPixel', 'numSatMeasurementsStrip', 'dRMinJet', 'ecalo', 'pt', 'd0', 'dz', 'totalCharge', 'deltaRToClosestElectron', 'deltaRToClosestMuon', 'deltaRToClosestTauHad', 'normalizedChi2', 
    'sumEnergy', 'diffEnergy', 'encodedLayers', 'dz1', 'dz2', 'dz3', 'd01', 'd02', 'd03',
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
           'pixelHitSizeY16', 'stripSelection16', 'hitPosX16', 'hitPosY16']

varDict = {}
for x in range(len(variables)):
    varDict[variables[x]] = x

class validatorArgs:

    def __init__(self, outputDir, dataDir, weightsDir):
        self.outputDir = outputDir
        self.dataDir = dataDir
        self.weightsDir = weightsDir

class Model():

    def __init__(self, filters, input_dim, batch_norm, metrics):
        self.filters = filters
        self.input_dim = input_dim
        self.batch_norm = batch_norm
        self.metrics = metrics

    def buildModel(self, filters = [16, 8], input_dim = 55, batch_norm = False, metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]):
        #begin NN model
        model = Sequential()
        model.add(Dense(filters[0], input_dim=input_dim, activation='relu'))
        for i in range(len(filters)-1):
            model.add(Dense(filters[i+1], activation='relu'))
            if(batch_norm): model.add(BatchNormalization())
            model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

    def callModel(self):
        model = buildModel(self.filters, self.input_dim, self.batch_norm)
        return model

def saveConfig(inputDict, filename):
    out_config = {}
    for key, value in inputDict.items():
       if key == 'val_metrics':
           out_config['val_metrics'] = [str(x) for x in value]
       else:
           out_config[key] = value 

    configs = json.dumps(out_config)
    with open(filename, 'w') as outfile:
        outfile.write(configs)

def readConfig(configFile):
    with open(configFile, 'r') as openfile:
        config = json.load(openfile)
    return config
    
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

def loadData(dataDir, undersample, inputs, normalize_data, saveCategories, train_size, val_size, DEBUG=False):
    # load the dataset
    file_count = 0
    realTracks = []
    fakeTracks = []
    pileupTracks = []
    if dataDir.endswith('.npz'):
        myfile = np.load(dataDir, allow_pickle=True)
        trainTracks = np.array(myfile['trainTracks'])
        testTracks = np.array(myfile['testTracks'])
        valTracks = np.array(myfile['valTracks'])
        trainTruth = np.array(myfile['trainTruth'])
        testTruth = np.array(myfile['testTruth'])
        valTruth = np.array(myfile['valTruth'])
        #if(saveCategories['fake'] == True): fakeTracks = np.array(myfile['fake_infos'])
        #if(saveCategories['real'] == True): realTracks = np.array(myfile['real_infos'])
        #if(saveCategories['pileup'] == True): pileupTracks = np.array(myfile['pileup_infos'])

        ##fakeTracks = selectInputs(fakeTracks, inputs)
        #realTracks = selectInputs(realTracks, inputs)
        #pileupTracks = selectInputs(pileupTracks, inputs)

        return trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth

    else:
        for filename in os.listdir(dataDir):
            print("Loading...", dataDir + filename)
            if(DEBUG): 
                if file_count > 50: break
            myfile = np.load(dataDir+filename, allow_pickle=True)
            if(saveCategories['fake'] == True):
                fakes = np.array(myfile["fake_infos"])
                if len(fakes) != 0:
                    fakes = selectInputs(fakes, inputs)
            if(saveCategories['real'] == True):
                reals = np.array(myfile["real_infos"])
                if len(reals) != 0: 
                    reals = selectInputs(reals, inputs)
            if(saveCategories['pileup'] == True):
                pileup = np.array(myfile["pileup_infos"])
                if len(pileup) != 0:
                    pileup = selectInputs(pileup, inputs)
            if(file_count == 0):
                if(saveCategories['fake'] == True): fakeTracks = fakes
                if(saveCategories['real'] == True): realTracks = reals
                if(saveCategories['pileup'] == True): pileupTracks = pileup
            elif(file_count != 0 and len(fakeTracks) == 0 and saveCategories['fake'] == True): fakeTracks = fakes
            elif(file_count != 0 and len(realTracks) == 0 and saveCategories['real'] == True): realTracks = reals
            elif(file_count != 0 and len(pileupTracks) == 0 and saveCategories['pileup'] == True): pileupTracks = pileup
            else:
                if(saveCategories['fake'] == True):
                    if(len(fakes)!=0): fakeTracks = np.concatenate((fakeTracks, fakes))
                if(saveCategories['real'] == True):
                    if(len(reals)!=0): realTracks = np.concatenate((realTracks, reals))
                if(saveCategories['pileup'] == True):
                    if(len(pileup)!=0): pileupTracks = np.concatenate((pileupTracks, pileup))
            file_count += 1


    print(bcolors.BLUE + "Number of fake tracks:", len(fakeTracks))
    print("Number of real tracks:", len(realTracks))
    print("Number of pileup tracks:", len(pileupTracks), bcolors.ENDC)

    if(saveCategories['real'] == True):
        if(train_size > 0):
            trainRealTracks, testRealTracks, trainRealTruth, testRealTruth = train_test_split(realTracks, np.zeros(len(realTracks)), test_size = 1-train_size)
        if(train_size == 0): 
            trainRealTracks = realTracks
            testRealTracks = []
            trainRealTruth = np.zeros(len(realTracks))
            testRealTruth = []
        if(val_size > 0):
            testRealTracks, valRealTracks, testRealTruth, valRealTruth = train_test_split(testRealTracks, testRealTruth, test_size = val_size)
        if(val_size == 0):
            valRealTracks = []
            valRealTruth = []
    if(saveCategories['fake'] == True):
        if(train_size > 0):
            trainFakeTracks, testFakeTracks, trainFakeTruth, testFakeTruth = train_test_split(fakeTracks, np.ones(len(fakeTracks)), test_size = 1-train_size)
        if(train_size == 0):
            trainFakeTracks = fakeTracks
            testFakeTracks = []
            trainFakeTruth = np.ones(len(fakeTracks))
            testFakeTruth = []
        if(val_size > 0):
            testFakeTracks, valFakeTracks, testFakeTruth, valFakeTruth = train_test_split(testFakeTracks, testFakeTruth, test_size = val_size)
        if(val_size == 0):
            valFakeTracks = []
            valFakeTruth = []
        #trainFakeTracks, testFakeTracks, trainFakeTruth, testFakeTruth = train_test_split(fakeTracks, np.ones(len(fakeTracks)), test_size = 1-train_size)
        #testFakeTracks, valFakeTracks, testFakeTruth, valFakeTruth = train_test_split(testFakeTracks, testFakeTruth, test_size = val_size)
     
    if(saveCategories['pileup'] == True):
        indices = np.arange(len(pileupTracks))
        np.random.shuffle(indices)
        pileupTracks = pileupTracks[indices]
        return pileupTracks, [], [], 2*np.ones(len(pileupTracks)), [], [] 
   
    # if undersampling
    if(undersample != -1 and saveCategories['real'] == True):
        num_real = len(trainRealTracks)
        num_select = int(undersample * num_real)
        ind = np.arange(num_real)
        ind = np.random.choice(ind, num_select)
        trainRealTracks = trainRealTracks[ind]


    #combine all data and shuffle
    if(saveCategories['real'] == True):
        if(saveCategories['fake'] == False or len(trainFakeTracks)==0):
            trainTracks = np.array(trainRealTracks)
            testTracks = np.array(testRealTracks)
            trainTruth = np.array(trainRealTruth)
            testTruth = np.array(testRealTruth)
            valTracks = np.array(valRealTracks)
            valTruth = np.array(valRealTruth)
        elif(saveCategories['fake'] == True):
            trainTracks = np.concatenate((trainFakeTracks, trainRealTracks))
            testTracks = np.concatenate((testFakeTracks, testRealTracks))
            trainTruth = np.concatenate((trainFakeTruth, trainRealTruth))
            testTruth = np.concatenate((testFakeTruth, testRealTruth))
            valTracks = np.concatenate((valFakeTracks, valRealTracks))
            valTruth = np.concatenate((valFakeTruth, valRealTruth))
    elif(saveCategories['real'] == False and saveCategories['fake'] == True):
        trainTracks = np.array(trainFakeTracks)
        testTracks = np.array(testFakeTracks)
        trainTruth = np.array(trainFakeTruth)
        testTruth = np.array(testFakeTruth)
        valTracks = np.array(valFakeTracks)
        valTruth = np.array(valFakeTruth)

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

def createSplitDataset():

    return 0

def loadSplitDataset(dataset, inputs):
    fin = np.load(dataset)
    trainTracks = fin['trainTracks']
    testTracks = fin['testTracks']
    valTracks = fin['valTracks']
    trainTruth = fin['trainTruth']
    testTruth = fin['testTruth']
    valTruth = fin['valTruth']

    trainTracks = selectInputs(trainTracks, inputs)
    testTracks = selectInputs(testTracks, inputs)
    valTracks = selectInputs(valTracks, inputs)

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
            #if y == 'eventNumber':
            #    print("Not deleting event number, need this for test")
            #    continue
            if y in x:
                #print(x, varDict[x])
                delete_array.append(varDict[x])
    inputs = np.delete(inputs, delete_array)
    input_dim = len(inputs) - 1 #event number is left in array for now
    return inputs, input_dim
    #return inputs, input_dim+1 #testing with old network

def createDict(variables):
    varDict = {}
    for x in range(len(variables)):
        varDict[variables[x]] = x

    return varDict

def listVariables(inputs):

    myinputs = variables
    myinputs = np.take(myinputs, inputs)
    return myinputs

if __name__ == '__main__':

    print(len(variables))

    for x in range(len(variables)):
        varDict[variables[x]] = x

    print(varDict)



    delete_elements = []
    undersample = -1
    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_DYJets_aMCNLO_v9p3/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_v9p3/"]
    input_dim = len(variables)
    normalize_data = False
    saveCategories = [{'fake':True, 'real':True, 'pileup':False}, {'fake':True, 'real':False, 'pileup':False}]
    train_size = 0.7
    val_size = 0.5

    inputs, input_dim = getInputs(input_dim, delete_elements)
    trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = loadData(dataDir[0], undersample, inputs, normalize_data, saveCategories[0], train_size, val_size, DEBUG=False)

    np.savez_compressed('fakeNNInputDataset_DYJets.npz', trainTracks = trainTracks, testTracks = testTracks, valTracks = valTracks, trainTruth = trainTruth, testTruth = testTruth, valTruth = valTruth)

    trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = loadData(dataDir[1], undersample, inputs, normalize_data, saveCategories[1], train_size, val_size, DEBUG=False)

    np.savez_compressed('fakeNNInputDataset_NG.npz', trainTracks = trainTracks, testTracks = testTracks, valTracks = valTracks, trainTruth = trainTruth, testTruth = testTruth, valTruth = valTruth)

    print("Train Tracks " + str(trainTracks.shape))
    print("Train Truth " + str(trainTruth.shape))
    print("Test Tracks " + str(testTracks.shape))
    print("Test Truth " + str(testTruth.shape))
    print("Val Tracks " + str(valTracks.shape))
    print("Val Truth " + str(valTruth.shape))

