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
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def buildModel(filters = [16, 8], input_dim = 55, batch_norm = False, metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]):
    #begin NN model
    model = Sequential()
    model.add(Dense(filters[0], input_dim=input_dim, activation='relu'))
    for i in range(len(filters)-1):
        model.add(Dense(filters[i+1], activation='relu'))
        if(batch_norm): model.add(BatchNormalization())
        model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    print(model.summary())
    return model

def callModel():
    model = buildModel(filters, input_dim, batch_norm)
    return model

def loadData(dataDir, undersample):
    #layerId, charge, isPixel, pixelHitSize, pixelHitSizeX, pixelHitSizeY, stripShapeSelection, hitPosX, hitPosY
    # load the dataset
    file_count = 0
    realTracks = []
    fakeTracks = []
    pileupTracks = []
    for filename in os.listdir(dataDir):
        print("Loading...", dataDir + filename)
        #if file_count > 20: break
        myfile = np.load(dataDir+filename)
        fakes = np.array(myfile["fake_infos"])
        reals = np.array(myfile["real_infos"])
        pileup = np.array(myfile["pileup_infos"])
        if(file_count == 0):
            fakeTracks = fakes
            realTracks = reals
            pileupTracks = pileup
        elif(file_count != 0 and len(fakeTracks) == 0): fakeTracks = fakes
        elif(file_count != 0 and len(realTracks) == 0): realTracks = reals
        elif(file_count != 0 and len(pileupTracks) == 0): pileupTracks = pileup
        else:
            if(len(fakes)!=0): fakeTracks = np.concatenate((fakeTracks, fakes))
            if(len(reals)!=0): realTracks = np.concatenate((realTracks, reals))
            if(len(pileup)!=0): pileupTracks = np.concatenate((pileupTracks, pileup))
        file_count += 1


    print("Number of fake tracks:", len(fakeTracks))
    print("Number of real tracks:", len(realTracks))
    print("Number of pileup tracks:", len(pileupTracks))

    trainRealTracks, testRealTracks, trainRealTruth, testRealTruth = train_test_split(realTracks, np.zeros(len(realTracks)), test_size = 0.3)
    trainFakeTracks, testFakeTracks, trainFakeTruth, testFakeTruth = train_test_split(fakeTracks, np.ones(len(fakeTracks)), test_size = 0.3)
    trainPileupTracks, testPileupTracks, trainPileupTruth, testPileupTruth = train_test_split(pileupTracks, 2*np.ones(len(pileupTracks)), test_size = 0.3)

    testRealTracks, valRealTracks, testRealTruth, valRealTruth = train_test_split(testRealTracks, testRealTruth, test_size = 0.5)
    testFakeTracks, valFakeTracks, testFakeTruth, valFakeTruth = train_test_split(testFakeTracks, testFakeTruth, test_size = 0.5)
    testPileupTracks, valPileupTracks, testPileupTruth, valPileupTruth = train_test_split(testPileupTracks, testPileupTruth, test_size = 0.5)


    # if undersampling
    if(undersample != -1):
        num_real = len(trainRealTracks)
        num_select = int(undersample * num_real)
        ind = np.arange(num_real)
        ind = np.random.choice(ind, num_select)
        trainRealTracks = trainRealTracks[ind]
        

    #combine all data and shuffle
    trainTracks = np.concatenate((trainFakeTracks, trainRealTracks, trainPileupTracks))
    testTracks = np.concatenate((testFakeTracks, testRealTracks, testPileupTracks))
    trainTruth = np.concatenate((trainFakeTruth, trainRealTruth, trainPileupTruth))
    testTruth = np.concatenate((testFakeTruth, testRealTruth, testPileupTruth))
    valTracks = np.concatenate((valFakeTracks, valRealTracks, valPileupTracks))
    valTruth = np.concatenate((valFakeTruth, valRealTruth, valPileupTruth))

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
    trainTracks = np.reshape(trainTracks, (-1,input_dim))
    if(normalize_data): trainTracks = np.tanh(trainTracks)
    trainTruth = trainTruth[train_indices]

    testTracks = testTracks[test_indices]
    testTracks = np.reshape(testTracks, (-1,input_dim))
    if(normalize_data): testTracks = np.tanh(testTracks)
    testTruth = testTruth[test_indices]

    valTracks = valTracks[val_indices]
    valTracks = np.reshape(valTracks, (-1,input_dim))
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
        print(plotMetrics.bcolors.RED+"USAGE: categoricalNN.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+plotMetrics.bcolors.ENDC)
        sys.exit(2)

    workDir = 'outcatNN_' + date.today().strftime('%m_%d')
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
            print(plotMetrics.bcolors.RED+"USAGE: categoricalNN.py -d/--dir= output_directory -p/--params= parameters.npy -i/--index= parameter_index"+plotMetrics.bcolors.ENDC)
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

    dataDir = ["/store/user/mcarrigan/fakeTracks/converted_aMC_cat0p1_4PlusLayer_v7p1/"]
    normalize_data = False
    undersample = -1
    oversample = -1   
    filters=[12,8]
    batch_norm = False
    batch_size = 64
    epochs = 5
    input_dim = 163
    patience_count = 20
    monitor = 'val_loss'
    #class_weights = False  
    metrics = [keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
    #################################################

    if(len(params) > 0):
        filters = params[0]
        batch_norm = bool(params[1])
        undersample = float(params[2])
        epochs = int(params[3])
        dataDir = params[4]
        input_dim = params[5]

    # create output directories
    os.system('mkdir '+str(workDir))
    os.system('mkdir '+str(plotDir))
    os.system('mkdir '+str(weightsDir))
    os.system('mkdir '+str(outputDir))
    
    for i, dataSet in enumerate(dataDir):
        if i == 0:
            trainTracks, testTracks, valTracks, trainTruth, testTruth, valTruth = loadData(str(dataSet), undersample)
        else:
            trainTracks2, testTracks2, valTracks2, trainTruth2, testTruth2, valTruth2 = loadData(str(dataSet), undersample)
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
    
    #one hot encode truth values
    encoder = LabelEncoder()
    encoder.fit(trainTruth)
    trainTruth = encoder.transform(trainTruth)
    testTruth = encoder.transform(testTruth)
    valTruth = encoder.transform(valTruth)
    trainTruth = np_utils.to_categorical(trainTruth)
    testTruth = np_utils.to_categorical(testTruth)
    valTruth = np_utils.to_categorical(valTruth)
     
    #Key for vector types
    vectors = np_utils.to_categorical(encoder.transform([0,1,2]))
    realVector = vectors[0]
    fakeVector = vectors[1]
    pileupVector = vectors[2]

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
    predictions = estimator.predict(testTracks)
    
    pred_real, true_real = 0, 0
    pred_fake, true_fake = 0, 0
    pred_pileup, true_pileup = 0, 0
    real_predictions, fake_predictions, pileup_predictions = np.zeros(3), np.zeros(3), np.zeros(3)
   
    for ipred, pred in enumerate(predictions):
        pred_index = -1
        this_true = testTruth[ipred]
        if(pred == 0): 
            pred_real+=1
            pred_index = 0
        elif(pred == 1): 
            pred_fake+=1
            pred_index = 2
        elif(pred == 2): 
            pred_pileup+=1 
            pred_index = 1
        if(this_true == realVector).all(): 
            true_real+=1
            real_predictions[pred_index] += 1
        elif(this_true == fakeVector).all(): 
            true_fake+=1
            fake_predictions[pred_index] += 1
        elif(this_true == pileupVector).all(): 
            true_pileup+=1
            pileup_predictions[pred_index] += 1

    all_predictions = [real_predictions, pileup_predictions, fake_predictions]

    print("Predicted Reals: "  +str(pred_real), " ... Truth Reals: " + str(true_real))
    print("Predicted Fakes: "  +str(pred_fake), " ... Truth Fakes: " + str(true_fake))
    print("Predicted Pileup: "  +str(pred_pileup), " ... Truth Pileup: " + str(true_pileup))

    print(all_predictions)
    plotMetrics.plotCM3(all_predictions, plotDir)
    #plotMetrics.getStats(testTruth, predictions)
    plotMetrics.plotHistory(history, ['loss', 'auc', 'recall', 'precision'], plotDir)
    #plotMetrics.permutationImportance(estimator, testTracks, testTruth, plotDir)
    np.savez_compressed(outputDir + "predictions.npz", tracks = testTracks, truth = testTruth, predictions = predictions)



