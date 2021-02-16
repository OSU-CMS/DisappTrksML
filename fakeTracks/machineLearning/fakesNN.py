import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
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

def buildModel(filters = [16, 8], metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]):
    #begin NN model
    #metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    #filters = [16, 8]
    model = Sequential()
    model.add(Dense(filters[0], input_dim=156, activation='relu'))
    for i in range(len(filters)-1):
        model.add(Dense(filters[i+1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    print(model.summary())
    return model

def callModel():
    model = buildModel(filters)
    return model

def buildSimple():
    model = keras.Sequential()
    model.add(keras.layers.Dense(12, input_dim=156, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])

    print(model.summary())
    return model

def create_baseline():
        # create model
        model = Sequential()
        model.add(Dense(60, input_dim=156, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        return model

def loadData(dataDir):
    #layerId, charge, isPixel, pixelHitSize, pixelHitSizeX, pixelHitSizeY, stripShapeSelection, hitPosX, hitPosY
    # load the dataset
    file_count = 0
    for filename in os.listdir(dataDir):
        print("Loading...", dataDir + filename)
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

    trainRealTracks, testRealTracks, trainRealTruth, testRealTruth = train_test_split(realTracks, np.zeros(len(realTracks)), test_size = 0.3)
    trainFakeTracks, testFakeTracks, trainFakeTruth, testFakeTruth = train_test_split(fakeTracks, np.ones(len(fakeTracks)), test_size = 0.3)

    testRealTracks, valRealTracks, testRealTruth, valRealTruth = train_test_split(testRealTracks, testRealTruth, test_size = 0.5)
    testFakeTracks, valFakeTracks, testFakeTruth, valFakeTruth = train_test_split(testFakeTracks, testFakeTruth, test_size = 0.5)

    #combine all data and shuffle
    trainTracks = np.concatenate((trainFakeTracks, trainRealTracks))
    testTracks = np.concatenate((testFakeTracks, testRealTracks))
    trainTruth = np.concatenate((trainFakeTruth, trainRealTruth))
    testTruth = np.concatenate((testFakeTruth, testRealTruth))
    valTracks = np.concatenate((valFakeTracks, valRealTracks))
    valTruth = np.concatenate((valFakeTruth, valRealTruth))

    test_indices = np.arange(len(testTracks))
    np.random.shuffle(test_indices)
    train_indices = np.arange(len(trainTracks))
    np.random.shuffle(train_indices)
    val_indices = np.arange(len(valTracks))
    np.random.shuffle(val_indices)

    trainTracks = trainTracks[train_indices]
    trainTracks = np.reshape(trainTracks, (-1,156))
    if(normalize_data): trainTracks = np.tanh(trainTracks)
    trainTruth = trainTruth[train_indices]

    testTracks = testTracks[test_indices]
    testTracks = np.reshape(testTracks, (-1,156))
    if(normalize_data): testTracks = np.tanh(testTracks)
    testTruth = testTruth[test_indices]

    valTracks = valTracks[val_indices]
    valTracks = np.reshape(valTracks, (-1,156))
    if(normalize_data): valTracks = np.tanh(valTracks)
    valTruth = valTruth[val_indices]

    return trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth


if __name__ == "__main__":

    # limit CPU usage
    config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads = 4,   
                                      intra_op_parallelism_threads = 4,
                                      allow_soft_placement = True,
                                      device_count={'CPU': 4})
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # suppress warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:p:i:", ["dir=","params=","index="])
    except getopt.GetoptError:
        print(plotMetrics.bcolors.RED+"USAGE: fakesNN.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+plotMetrics.bcolors.ENDC)
        sys.exit(2)

    workDir = 'fakesNN_' + date.today().strftime('%m_%d')
    print("workDir", workDir)
    paramsFile = ""
    params = []
    paramsIndex = 0
    for opt, arg in opts:
        if(opt in ('-d','--dir')):
            workDir = str(arg)
        elif(opt in ('-p','--params')):
            paramsFile = str(arg)
        elif(opt in ('-i','--index')):
            paramsIndex = int(arg)

    if(len(paramsFile)>0):
        try:
            params = np.load(str(paramsFile), allow_pickle=True)[paramsIndex]
        except:
            print(plotMetrics.bcolors.RED+"ERROR: Index outside range or no parameter list passed"+plotMetrics.bcolors.ENDC)
            print(plotMetrics.bcolors.RED+"USAGE: fakesNN.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+plotMetrics.bcolors.ENDC)
            sys.exit(2)
        workDir = workDir + "_p" + str(paramsIndex)
    cnt=0
    while(os.path.isdir(workDir)):
        cnt+=1
        if(cnt==1): workDir = workDir+"_"+str(cnt)
        else: workDir = workDir[:-1] + str(cnt)
    print(plotMetrics.bcolors.YELLOW+"Output directory: "+workDir+plotMetrics.bcolors.ENDC)
    if(len(params) > 0): 
        print(plotMetrics.bcolors.YELLOW+"Using params"+plotMetrics.bcolors.ENDC, params, ' ')
        print(plotMetrics.bcolors.YELLOW+"from file "+paramsFile+plotMetrics.bcolors.ENDC)
	
    plotDir = workDir + '/plots/'
    weightsDir = workDir + '/weights/'
    outputDir = workDir + '/outputFiles/'

    ################config parameters################
    """
    nTotE: number of electron events to use
    oversample_e: (NOT WORKING) fraction of electron events per train batch, set to -1 if it's not needed
    undersample_bkg: fraction of backgruond events per train batch, set to -1 if it's not needed
    v: verbosity
    patience_count: after how many epochs to stop if monitored variable doesn't improve
    monitor: which variable to monitor with patience_count
    """

    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_v1/", "/store/user/mcarrigan/fakeTracks/converted_aMC_v1/"]
    #logDir = "/home/llavezzo/work/cms/logs/"+ workDir +"_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    normalize_data = False
    #run_validate = True
    #nTotE = 27000
    #val_size = 0.4
    #undersample_bkg = -1
    #oversample_e = -1   
    filters=[128,32]
    #batch_norm = False
    #v = 2
    batch_size = 32
    epochs = 10
    #patience_count = 20
    #monitor = 'val_loss'
    #class_weights = False  
    metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    #################################################

    if(len(params) > 0):
        filters = params[0]
        class_weights = bool(params[1])
        undersample_bkg = float(params[2])
        epochs = int(params[3])
        dataDir = params[4]

    # create output directories
    os.system('mkdir '+str(workDir))
    os.system('mkdir '+str(plotDir))
    os.system('mkdir '+str(weightsDir))
    os.system('mkdir '+str(outputDir))
    
    for i, dataSet in enumerate(dataDir):
        if i == 0:
            trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = loadData(str(dataSet))
        else:
            trainTracks2, testTracks2, valTracks2, trainTruth2, testTruth2, valTruth2 = loadData(str(dataSet))
            trainTracks = np.concatenate((trainTracks, trainTracks2))
            trainTruth = np.concatenate((trainTruth, trainTruth2))
            testTracks = np.concatenate((testTracks, testTracks2))
            testTruth = np.concatenate((testTruth, testTruth2))
            valTracks = np.concatenate((valTracks, valTracks2))
            valTruth = np.concatenate((valTruth, valTruth2))

    print("Train Tracks " + str(trainTracks.shape))
    print("Train Truth " + str(trainTruth.shape))
    print("Test Tracks " + str(testTracks.shape))
    print("Test Truth " + str(testTruth.shape))
    print("Val Tracks " + str(valTracks.shape))
    print("Val Truth " + str(valTruth.shape))

    indices = np.arange(len(trainTracks))
    np.random.shuffle(indices)   
    trainTracks = trainTracks[indices]
    trainTruth = trainTruth[indices]
    
    indices = np.arange(len(testTracks))
    np.random.shuffle(indices)
    testTracks = testTracks[indices]
    testTruth = testTruth[indices]

    indices = np.arange(len(valTracks))
    np.random.shuffle(indices)
    valTracks = valTracks[indices]
    valTruth = valTruth[indices]         
    
    model = callModel

    estimator = KerasClassifier(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=1)
    history = estimator.fit(trainTracks, trainTruth, validation_data=(valTracks, valTruth))

    predictions = estimator.predict(testTracks)

    plotMetrics.plotCM(testTruth, predictions, plotDir)
    plotMetrics.getStats(testTruth, predictions)
    plotMetrics.plotHistory(history, ['loss', 'auc', 'recall', 'precision'], plotDir)
    plotMetrics.permutationImportance(estimator, testTracks, testTruth, plotDir)
    estimator.model.save_weights(weightsDir+'lastEpoch.h5')
    np.savez_compressed(outputDir + "predictions.npz", tracks = testTracks, truth = testTruth, predictions = predictions)



