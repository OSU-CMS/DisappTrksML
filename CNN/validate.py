import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import random
import sys
import pickle
import utils
import flow as cnn

def validate(model, batchDir, dataDir, tag, plotDir):

	# load the batches used to train and validate
    val_e_file_batches = np.load(batchDir+'e_files_valBatches.npy')
    val_e_event_batches = np.load(batchDir+'e_events_valBatches.npy')
    val_bkg_file_batches = np.load(batchDir+'bkg_files_valBatches.npy')
    val_bkg_event_batches = np.load(batchDir+'bkg_events_valBatches.npy')

    validatedE, validatedBkg = 0,0
    iBatch=0
    for files, indices in zip(val_e_file_batches, val_e_event_batches):
        lastFile = len(files)-1
        files.sort()
        for iFile, file in enumerate(files):
            if(file == -1): 
                e_images = np.array([])
                continue

            if(iFile == 0 and iFile != lastFile):
                e_images = np.load(dataDir+'e_'+tag+str(file)+'.npy')[indices[0]:]

            elif(iFile == lastFile and iFile != 0):
                e_images = np.concatenate((e_images,np.load(dataDir+'e_'+tag+str(file)+'.npy')[:indices[1]+1]))

            elif(iFile == 0 and iFile == lastFile):
                e_images = np.load(dataDir+'e_'+tag+str(file)+'.npy')[indices[0]:indices[1]+1]

            elif(iFile != 0 and iFile != lastFile):
                e_images = np.concatenate((e_images,np.load(dataDir+'e_'+tag+str(file)+'.npy')))
            
        e_images = np.reshape(e_images[:,1:],(e_images.shape[0],40,40,4))
        e_images = e_images[:,:,:,[0,2,3]]
        validatedE+=e_images.shape[0]
        if(iBatch==0): predictionsE = model.predict(e_images)
        else: predictionsE = np.concatenate([predictionsE, model.predict(e_images)])
        iBatch+=1

    iBatch=0
    for files, indices in zip(val_bkg_file_batches, val_bkg_event_batches):
        lastFile = len(files)-1
        files.sort()
        for iFile, file in enumerate(files):
            if(iFile == 0 and iFile != lastFile):
                bkg_images = np.load(dataDir+'bkg_'+tag+str(file)+'.npy')[indices[0]:,:]

            elif(iFile == lastFile and iFile != 0):
                bkg_images = np.concatenate((bkg_images,np.load(dataDir+'bkg_'+tag+str(file)+'.npy')[:indices[1]+1,:]))

            elif(iFile == 0 and iFile == lastFile):
                bkg_images = np.load(dataDir+'bkg_'+tag+str(file)+'.npy')[indices[0]:indices[1]+1,:]

            elif(iFile != 0 and iFile != lastFile):
                bkg_images = np.concatenate((bkg_images,np.load(dataDir+'bkg_'+tag+str(file)+'.npy')))
        
        bkg_images = np.reshape(bkg_images[:,1:],(bkg_images.shape[0],40,40,4))
        bkg_images = bkg_images[:,:,:,[0,2,3]]
        validatedBkg+=bkg_images.shape[0]
        if(iBatch==0): predictionsB = model.predict(bkg_images)
        else: predictionsB = np.concatenate([predictionsB, model.predict(bkg_images)])
        iBatch+=1
    
                
    predictions = np.concatenate((predictionsE, predictionsB))
    true = np.concatenate((np.ones(len(predictionsE)), np.zeros(len(predictionsB))))
    #y_test = keras.utils.to_categorical(true, num_classes=2)

    print(predictions)
    sys.exit(0)
    
    utils.metrics(true, predictions, plotDir, threshold=0.5)

    print()
    print(utils.bcolors.HEADER+"Validated on "+str(validatedE)+" electron events and "+str(validatedBkg)+" background events"+utils.bcolors.ENDC)
    print(utils.bcolors.GREEN+"Saved metrics to "+plotDir+utils.bcolors.ENDC)
    print()


if __name__ == "__main__":

	dataDir = "/store/user/llavezzo/disappearingTracks/electron_selectionV2/"
	batchDir = "cnn_results/outputFiles/"
	plotDir = "cnn_results/plots/"
	valFile = "valBatches"
	tag = "0p25_tanh_"

	input_shape = (40,40,3)

	model = cnn.build_model(input_shape = input_shape, 
						layers = 5, filters = 64, opt='adam')

	validate(model, batchDir, dataDir, tag, plotDir)