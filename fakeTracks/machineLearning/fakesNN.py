import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
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
from fakeClass import fakeNN
import cmsml

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
        opts, args = getopt.getopt(sys.argv[1:], "d:p:i:g:", ["dir=","params=","index=", "grid="])
    except getopt.GetoptError:
        print(plotMetrics.bcolors.RED+"USAGE: fakesNN.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index -g/--grid= grid_search"+plotMetrics.bcolors.ENDC)
        sys.exit(2)

    workDir = 'outfakesNN_' + date.today().strftime('%m_%d')
    print("workDir", workDir)
    paramsFile = ""
    params = []
    paramsIndex = 0
    gridSearch = -1
    for opt, arg in opts:
        if(opt in ('-d','--dir')):
            workDir = str(arg)
        elif(opt in ('-p','--params')):
            paramsFile = str(arg)
        elif(opt in ('-i','--index')):
            paramsIndex = int(arg)
        elif(opt in ('-g', '--grid')):
            gridSearch = int(arg)

    if(len(paramsFile)>0):
        try:
            params = np.load(str(paramsFile), allow_pickle=True)[paramsIndex]
        except:
            print(plotMetrics.bcolors.RED+"ERROR: Index outside range or no parameter list passed"+plotMetrics.bcolors.ENDC)
            print(plotMetrics.bcolors.RED+"USAGE: fakesNN.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+plotMetrics.bcolors.ENDC)
            sys.exit(2)
        if(gridSearch > 0):
            workDir = workDir + "_g" + str(gridSearch) + "_p" + str(paramsIndex)
        else:
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

    DEBUG = True

    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_DYJets_aMCNLO_v9p3/", "/store/user/mcarrigan/fakeTracks/converted_NeutrinoGun_ext_v9p3/"]
    #dataDir = ['fakeNNInputDataset_DYJets.npz', 'fakeNNInputDataset_NG.npz']
    normalize_data = False
    undersample = -1
    oversample = -1   
    filters=[12,8]
    batch_norm = False
    batch_size = 64
    epochs = 10
    input_dim = 178
    dropout = 0.1
    patience_count = 20
    monitor = 'val_loss'
    #class_weights = False  
    val_metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    #delete_elements = ['totalCharge', 'numSatMeasurements', 'stripSelection', 'hitPosX', 'hitPosY', 'numMeasurementsPixel', 'layer', 'subDet']
    delete_elements = ['eventNumber', 'layer1', 'subDet1', 'layer2', 'subDet2','layer3', 'subDet3', 'layer4', 'subDet4', 'layer5', 'subDet5', 'layer6', 'subDet6', 'layer7', 'subDet7','layer8', 'subDet8', 'layer9', 'subDet9', 'layer10', 'subDet10', 'layer11', 'subDet11', 'layer12', 'subDet12', 'layer13', 'subDet13', 'layer15', 'subDet14', 'layer15', 'subDet15','layer16', 'subDet16']
    #delete_elements = ['passesSelection', 'eventNumber', 'dEdxPixel', 'dEdxStrip', 'numMeasurementsPixel', 'numMeasurementsStrip', 'numSatMeasurementsPixel', 'numSatMeasurementsStrip', 'totalCharge', 'deltaRToClosestElectron', 'deltaRToClosestMuon', 'deltaRToClosestTauHad', 'normalizedChi2', 'layer1', 'charge1', 'subDet1', 'pixelHitSize1', 'pixelHitSizeX1', 'pixelHitSizeY1','stripSelection1', 'hitPosX1', 'hitPosY1', 'layer2', 'charge2', 'subDet2', 'pixelHitSize2', 'pixelHitSizeX2', 'pixelHitSizeY2', 'stripSelection2', 'hitPosX2', 'hitPosY2', 'layer3', 'charge3', 'subDet3', 'pixelHitSize3', 'pixelHitSizeX3', 'pixelHitSizeY3', 'stripSelection3', 'hitPosX3', 'hitPosY3', 'layer4', 'charge4', 'subDet4', 'pixelHitSize4', 'pixelHitSizeX4', 'pixelHitSizeY4', 'stripSelection4', 'hitPosX4', 'hitPosY4', 'layer5', 'charge5', 'subDet5', 'pixelHitSize5', 'pixelHitSizeX5', 'pixelHitSizeY5', 'stripSelection5', 'hitPosX5', 'hitPosY5', 'layer6', 'charge6', 'subDet6', 'pixelHitSize6', 'pixelHitSizeX6', 'pixelHitSizeY6', 'stripSelection6', 'hitPosX6', 'hitPosY6', 'layer7', 'charge7', 'subDet7', 'pixelHitSize7', 'pixelHitSizeX7', 'pixelHitSizeY7', 'stripSelection7', 'hitPosX7', 'hitPosY7', 'layer8', 'charge8', 'subDet8', 'pixelHitSize8', 'pixelHitSizeX8', 'pixelHitSizeY8', 'stripSelection8', 'hitPosX8', 'hitPosY8', 'layer9', 'charge9', 'subDet9', 'pixelHitSize9', 'pixelHitSizeX9', 'pixelHitSizeY9', 'stripSelection9', 'hitPosX9', 'hitPosY9', 'layer10', 'charge10', 'subDet10', 'pixelHitSize10', 'pixelHitSizeX10', 'pixelHitSizeY10', 'stripSelection10', 'hitPosX10', 'hitPosY10', 'layer11', 'charge11', 'subDet11', 'pixelHitSize11', 'pixelHitSizeX11', 'pixelHitSizeY11', 'stripSelection11', 'hitPosX11', 'hitPosY11', 'layer12', 'charge12', 'subDet12', 'pixelHitSize12', 'pixelHitSizeX12','pixelHitSizeY12', 'stripSelection12', 'hitPosX12', 'hitPosY12', 'layer13', 'charge13', 'subDet13', 'pixelHitSize13', 'pixelHitSizeX13', 'pixelHitSizeY13', 'stripSelection13', 'hitPosX13', 'hitPosY13', 'layer14', 'charge14', 'subDet14', 'pixelHitSize14', 'pixelHitSizeX14', 'pixelHitSizeY14', 'stripSelection14', 'hitPosX14', 'hitPosY14', 'layer15', 'charge15', 'subDet15', 'pixelHitSize15', 'pixelHitSizeX15', 'pixelHitSizeY15', 'stripSelection15', 'hitPosX15', 'hitPosY15', 'layer16', 'charge16', 'subDet16', 'pixelHitSize16', 'pixelHitSizeX16', 'pixelHitSizeY16', 'stripSelection16', 'hitPosX16', 'hitPosY16', 'sumEnergy', 'diffEnergy', 'dz1', 'd01', 'dz2', 'd02', 'dz3', 'd03']
    saveCategories = [{'fake':True, 'real':True, 'pileup':False}, {'fake':True, 'real':False, 'pileup':False}]
    trainPCT = 0.7
    valPCT = 0.5
    loadSplitDataset = False

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
        trainPCT = params[8]
        valPCT = params[9]
        loadSplitDataset = params[10]
        dropout = params[11]

    #make sure event number is not input to network
    if 'eventNumber' not in delete_elements:
        delete_elements.append('eventNumber')

    # create output directories
    if not os.path.isdir(workDir):
        os.system('mkdir '+str(workDir))
    if not os.path.isdir(plotDir):
        os.system('mkdir '+str(plotDir))
    if not os.path.isdir(weightsDir):
        os.system('mkdir '+str(weightsDir))
    if not os.path.isdir(outputDir):
        os.system('mkdir '+str(outputDir))
   
    inputs, input_dim = utilities.getInputs(input_dim, delete_elements)
 
    for i, dataSet in enumerate(dataDir):
        if i == 0:
            if(loadSplitDataset):
                trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = utilities.loadSplitDataset(str(dataSet), inputs)
            else:
                trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = utilities.loadData(str(dataSet), undersample, inputs, normalize_data, saveCategories[i], trainPCT, valPCT, DEBUG)

        else:
            if(loadSplitDataset):
                trainTracks2, testTracks2, valTracks2, trainTruth2, testTruth2, valTruth2 = utilities.loadSplitDataset(str(dataSet), inputs)
            else:
                trainTracks2, testTracks2, valTracks2, trainTruth2, testTruth2, valTruth2 = utilities.loadData(str(dataSet), undersample, inputs, normalize_data, saveCategories[i], trainPCT, valPCT, DEBUG)
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
    
    callbacks = [keras.callbacks.EarlyStopping(patience=patience_count), keras.callbacks.ModelCheckpoint(filepath=weightsDir+'model.{epoch}.h5', save_best_only=True, monitor=monitor, mode='auto')]

    model = fakeNN(filters, input_dim, batch_norm, val_metrics, dropout)

    estimator = KerasClassifier(build_fn=model, epochs=epochs, batch_size=batch_size, verbose=1)
    history = estimator.fit(trainTracks, trainTruth, validation_data=(valTracks, valTruth), callbacks = callbacks)
 
    cmsml.tensorflow.save_graph("graph.pb", estimator.model, variables_to_constants=True)


    print(history.history.keys())

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
    classifications = plotMetrics.getStats(testTruth, predictions, plotDir, plot=True)
    plotMetrics.plotHistory(history, ['loss', 'auc_1', 'recall_1', 'precision_1'], plotDir)
    plotMetrics.plotScores(predictions_raw, testTruth, 'fakeNN', plotDir)
    plotMetrics.predictionThreshold(predictions_raw, testTruth, plotDir)
    #plotMetrics.permutationImportance(estimator, testTracks, testTruth, plotDir)
    
    inputs = utilities.listVariables(inputs)    

    np.savez_compressed(outputDir + "predictions.npz", tracks = testTracks, truth = testTruth, predictions = predictions, predictionScores = predictions_raw, inputs = inputs)

    fout = open(outputDir + 'networkInfo.txt', 'w')
    fout.write('Datasets: ' + str(dataDir) + '\nFilters: ' + str(filters) + '\nBatch Size: ' + str(batch_size) + '\nBatch Norm: ' + str(batch_norm) +  '\nInput Dim: ' + str(input_dim) + '\nPatience Count: ' + str(patience_count) + '\nMetrics: ' + str(val_metrics) + '\nDeleted Elements: ' + str(delete_elements) + '\nSaved Tracks: ' + str(saveCategories) + '\nTrain Percentage: ' + str(trainPCT) + '\nVal Percentage: ' + str(valPCT) + '\nTotal Epochs: ' + str(max_epoch) + '\nDropout: ' + str(dropout) + '\nMetrics: TP = %d, FP = %d, TN = %d, FN = %d' % (classifications[0], classifications[1], classifications[2], classifications[3]) + '\nPrecision: ' + str(classifications[4]) + '\nRecall: ' + str(classifications[5]))
    fout.close()
