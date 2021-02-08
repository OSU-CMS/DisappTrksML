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
from datetime import date

def buildModel(filters, metrics):
    #begin NN model
    model = keras.Sequential()
    model.add(keras.layers.Dense(filters[0], input_shape=(156,), activation='relu'))
    for i in range(len(filters)-1):
        model.add(keras.layers.Dense(filters[i+1], activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=metrics)

    print(model.summary())
    return model

def loadData(dataDir):
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
    return trainTracks, testTracks, trainTruth, testTruth


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

    workDir = 'fakesNN' + date.today().strftime('%m_%d')
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

    dataDir = "/store/user/mcarrigan/fakeTracks/converted_v1/"
    #logDir = "/home/llavezzo/work/cms/logs/"+ workDir +"_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    #run_validate = True
    #nTotE = 27000
    #val_size = 0.4
    #undersample_bkg = -1
    #oversample_e = -1   
    filters=[12,8]
    #batch_norm = False
    #v = 2
    batch_size = 256
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
        dataDir = str(params[4])

    # create output directories
    os.system('mkdir '+str(workDir))
    os.system('mkdir '+str(plotDir))
    os.system('mkdir '+str(weightsDir))
    os.system('mkdir '+str(outputDir))
    
    trainTracks, testTracks, trainTruth, testTruth = loadData(dataDir)
 
    model = buildModel(filters, metrics)

    # fit the keras model on the dataset
    history = model.fit(trainTracks, trainTruth, epochs=epochs, batch_size=batch_size, verbose=1)

    # make class predictions with the model
    predictions = model.predict_classes(testTracks)

    plotMetrics.plotCM(testTruth, predictions, plotDir)
    plotMetrics.getStats(testTruth, predictions)
    plotMetrics.plotHistory(history, ['loss','auc', 'recall', 'precision'], plotDir)

    model.save_weights(weightsDir+'lastEpoch.h5')
    np.savez_compressed(outputDir + "predictions.npz", tracks = testTracks, truth = testTruth, predictions = predictions)



