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
from sklearn.preprocessing import MinMaxScaler
import utilities

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

    workDir = 'outfakesNN_' + date.today().strftime('%m_%d')
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

    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_v9_DYJets_aMCNLO_4PlusLayer_v9p1/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_4PlusLayer_v9p1/"]
    normalize_data = False
    undersample = -1
    oversample = -1   
    filters=[12,8]
    batch_norm = False
    batch_size = 64
    epochs = 1
    input_dim = 173
    patience_count = 20
    monitor = 'val_loss'
    #class_weights = False  
    metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    delete_elements = ['totalCharge', 'numSatMeasurements', 'stripSelection', 'hitPosX', 'hitPosY', 'numMeasurementsPixel', 'layer', 'subDet']
    saveCategories = [{'fake':True, 'real':True, 'pileup':False}, {'fake':True, 'real':False, 'pileup':False}]

    #################################################

    if(len(params) > 0):
        filters = params[0]
        batch_norm = bool(params[1])
        undersample = float(params[2])
        epochs = int(params[3])
        dataDir = params[4]
        input_dim = params[5]
        delete_elements = params[6]
        saveCategories = params[7]

    # create output directories
    os.system('mkdir '+str(workDir))
    os.system('mkdir '+str(plotDir))
    os.system('mkdir '+str(weightsDir))
    os.system('mkdir '+str(outputDir))
   
    inputs, input_dim = utilities.getInputs(input_dim, delete_elements)
 
    for i, dataSet in enumerate(dataDir):
        if i == 0:
            trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = utilities.loadData(str(dataSet), undersample, inputs, normalize_data, saveCategories[i], 0.7, 0.5)
        else:
            trainTracks2, testTracks2, valTracks2, trainTruth2, testTruth2, valTruth2 = utilities.loadData(str(dataSet), undersample, inputs, normalize_data, saveCategories[i], 0.7, 0.5)
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
    
    callbacks = [keras.callbacks.EarlyStopping(patience=patience_count),                                                                                                                            keras.callbacks.ModelCheckpoint(filepath=weightsDir+'model.{epoch}.h5',                                                                                                            save_best_only=True,                                                                                                                                                               monitor=monitor,                                                                                                                                                                   mode='auto')]

    model = callModel

    estimator = KerasClassifier(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=1)
    history = estimator.fit(trainTracks, trainTruth, validation_data=(valTracks, valTruth), callbacks = callbacks)
 
    estimator.model.save_weights(weightsDir+'lastEpoch.h5')

    max_epoch = -1
    final_weights = 'lastEpoch.h5'
    for filename in os.listdir(weightsDir):
        if ".h5" and "model" in filename:
            this_epoch = filename.split(".")[1]
            if int(this_epoch) > max_epoch: 
                final_weights = 'model.' + this_epoch + '.h5'
                max_epoch = int(this_epoch)
    
    print('Loading weights...' + final_weights)
    estimator.model.load_weights(weightsDir + final_weights)
    predictions_raw = estimator.predict_proba(testTracks)
    predictions = estimator.predict(testTracks)
    predictions_raw = predictions_raw[:, 1]
    #predictions = [1 if x >= 0.5 else 0 for x in predictions_raw]
    plotMetrics.plotCM(testTruth, predictions, plotDir)
    plotMetrics.getStats(testTruth, predictions)
    plotMetrics.plotHistory(history, ['loss', 'auc', 'recall', 'precision'], plotDir)
    plotMetrics.plotScores(predictions_raw, testTruth, 'fakeNN', plotDir)
    #plotMetrics.permutationImportance(estimator, testTracks, testTruth, plotDir)
    np.savez_compressed(outputDir + "predictions.npz", tracks = testTracks, truth = testTruth, predictions = predictions, predictionScores = predictions_raw)



