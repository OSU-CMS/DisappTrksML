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

def validate(model, valFile, batchDir, dataDir, tag, endtag, plotDir):

	# load the batches used to train and validate
    val_e_file_batches = np.load(batchDir+'e_files_'+valFile+'.npy')
    val_e_event_batches = np.load(batchDir+'e_events_'+valFile+'.npy')
    val_bkg_file_batches = np.load(batchDir+'bkg_files_'+valFile+'.npy')
    val_bkg_event_batches = np.load(batchDir+'bkg_events_'+valFile+'.npy')

    validatedE, validatedBkg = 0,0
    iBatch=0
    
    all_indicesE = []
    all_indicesB = []
    for files, indices in zip(val_e_file_batches, val_e_event_batches):
        lastFile = len(files)-1
        files.sort()
        for iFile, file in enumerate(files):
            if(file == -1): 
                e_images = []
                continue
            e_file = np.load(dataDir+'e_'+tag+str(file)+endtag+'.npz')
            if(iFile == 0 and iFile != lastFile):
                e_images = np.array(e_file['images'])
                e_indices = np.arange(indices[0], e_images.shape[0])
                e_images = e_images[indices[0]:]
                e_indices = np.append(e_indices, -1)
            elif(iFile == lastFile and iFile != 0):
                e_temp = np.array(e_file['images'])
                e_images = np.concatenate((e_images, e_temp[:indices[1]+1]))
                e_indices = np.concatenate((e_indices, np.arange(indices[1]+1)))
                e_indices = np.append(e_indices, -1)
            elif(iFile == 0 and iFile == lastFile):
                e_images = np.array(e_file['images'])
                e_images = e_images[indices[0]:indices[1]+1]
                e_indices = np.arange(indices[0], indices[1]+1)
                e_indices = np.append(e_indices, -1)
            elif(iFile != 0 and iFile != lastFile):
                e_temp = np.array(e_file['images'])
                e_images = np.concatenate((e_images, e_temp))
                e_indices = np.concatenate((e_indices, np.arange(e_temp.shape[0])))
                e_indices = np.append(e_indices, -1)

        e_images = np.array(e_images)
        #print(e_images.shape)
        if e_images.shape[0] != 0:
            #temp_indices = e_images[:, 0]
            file_counter = 0
            #print("total events ", len(e_indices))
            #print("indices ", e_indices)
            for evts in range(len(e_indices)):
                #print("event num ", evts)
                if evts == 0: current_files = files
                if current_files != files: file_counter = 0
                if evts != 0:
                    if e_indices[evts] == -1:               
                        if len(files) > 1: file_counter += 1
                        continue
                    #print("temp ind: ", temp_indices[evts], temp_indices[evts-1])
                    #print("files and counter: ", files, file_counter)
                    current_file = int(files[file_counter])
                    this_index = [int(current_file), int(e_indices[evts]), 1]
                else: this_index = [int(file), int(e_indices[evts]), 1]
                all_indicesE.append(this_index)
            e_images = e_images[:, 1:]
            e_images = np.reshape(e_images,(e_images.shape[0],40,40,4))
            e_images = e_images[:, :, :, [0,2,3]]
            validatedE+=e_images.shape[0]
            if(iBatch==0): 
                predictionsE = model.predict(e_images)
                #e_indices = temp_indices
            else: 
                predictionsE = np.concatenate([predictionsE, model.predict(e_images)])
                #e_indices = np.concatenate([e_indices, temp_indices])
        iBatch+=1

    iBatch=0
    for files, indices in zip(val_bkg_file_batches, val_bkg_event_batches):
        lastFile = len(files)-1
        files.sort()
        for iFile, file in enumerate(files):
            if(iFile == 0 and iFile != lastFile):
                bkg_images = np.array(np.load(dataDir+'bkg_'+tag+str(file)+endtag+'.npz')['images'])
                bkg_indices = np.arange(indices[0], len(bkg_images))                
                #print("First File total, ", len(bkg_images))
                bkg_images = bkg_images[indices[0]:]
                #print("First File indices length ", len(bkg_indices))
                bkg_indices = np.append(bkg_indices, -1)
            elif(iFile == lastFile and iFile != 0):
                bkg_temp = np.array(np.load(dataDir+'bkg_'+tag+str(file)+endtag+'.npz')['images'])
                bkg_images = np.concatenate((bkg_images, bkg_temp[:indices[1]+1]))
                #print("indices before ", len(bkg_indices))
                bkg_indices = np.concatenate((bkg_indices, np.arange(indices[1]+1)))
                bkg_indices = np.append(bkg_indices, -1)
                #print("second file used ", len(bkg_indices), len(bkg_temp[:indices[1]+1]))
            elif(iFile == 0 and iFile == lastFile):
                bkg_images = np.array(np.load(dataDir+'bkg_'+tag+str(file)+endtag+'.npz')['images'])
                bkg_images = bkg_images[indices[0]:indices[1]+1]
                bkg_indices = np.arange(indices[0], indices[1]+1)
                bkg_indices = np.append(bkg_indices, -1)
            elif(iFile != 0 and iFile != lastFile):
                bkg_temp = np.array(np.load(dataDir+'bkg_'+tag+str(file)+endtag+'.npz')['images'])
                bkg_images = np.concatenate((bkg_images, bkg_temp))
                bkg_indices = np.concatenate((bkg_indices, np.arange(bkg_temp.shape[0])))
                bkg_indices = np.append(bkg_indices, -1)

        bkg_images = np.array(bkg_images)
        file_counter = 0
        #print("Total background events ", len(bkg_indices))
        for evts in range(len(bkg_indices)):
            #print("Event number ", evts, indices)
            if evts == 0: current_files = files
            if current_files != files: file_counter = 0
            print("bkg check ", evts, int(file), int(bkg_indices[evts]), files, file_counter)
            if evts != 0:
                if bkg_indices[evts] == -1:
                    if len(files) > 1: file_counter += 1
                    continue
                print("files and counter: ", files, file_counter)
                current_file = int(files[file_counter])
                this_index = [int(current_file), int(bkg_indices[evts]), 0]
            else: this_index = [int(file), int(bkg_indices[evts]), 0]
            if bkg_indices[evts] == -1: continue
            #print("bkg check ", evts, int(file), int(bkg_indices[evts]), files, file_counter)
            all_indicesB.append(this_index)
            current_files = files
        #print(len(all_indicesB))
        bkg_images = np.reshape(bkg_images[:,1:],(bkg_images.shape[0],40,40,4))
        bkg_images = bkg_images[:, :, :, [0,2,3]]
        validatedBkg+=bkg_images.shape[0]
        if(iBatch==0): 
            predictionsB = model.predict(bkg_images)
            #bkg_indices = temp_indices
        else: 
            predictionsB = np.concatenate([predictionsB, model.predict(bkg_images)])
            #bkg_indices = np.concatenate([bkg_indices, temp_indices])
        iBatch+=1
    
                
    predictions = np.concatenate((predictionsE, predictionsB))
    true = np.concatenate((np.ones(len(predictionsE)), np.zeros(len(predictionsB))))
    #y_test = keras.utils.to_categorical(true, num_classes=2)
    
    #e_false = utils.falsePrediction(predictionsE, np.ones(len(predictionsE)), e_indices)
    #bkg_false = utils.falsePrediction(predictionsB, np.zeros(len(predictionsB)), bkg_indices)
    
    #print("Total background events ", len(all_indicesB))

    firstWrong = 0
    eOut = np.array([])
    for ievent in range(len(all_indicesE)):
        pred = 0
        if predictionsE[int(ievent)][0] >= 0.5: 
            pred = 1
            firstWrong += 1
        print("electron prediction: ", pred, " from ", predictionsE[int(ievent)][0])
        if all_indicesE[int(ievent)][2] == pred: continue
        #if firstWrong == len(all_indicesE) -1: 
        #    eOut = np.array((-1, -1, -1, -1))
        #    continue
        print("first wrong ", firstWrong, "ievent ", ievent)
        line = np.array((all_indicesE[int(ievent)][0], all_indicesE[int(ievent)][1], all_indicesE[int(ievent)][2], predictionsE[int(ievent)][0]))
        if ievent == firstWrong: eOut = line
        else: eOut = np.concatenate((eOut, line))
    
    firstWrong = 0
    bkgOut = np.array([])
    for ievent in range(len(all_indicesB)):
        pred = 1
        if predictionsB[int(ievent)][0] < 0.5: 
            pred = 0
            firstWrong += 1
        if all_indicesB[int(ievent)][2] == pred: continue
        line = np.array((all_indicesB[int(ievent)][0], all_indicesB[int(ievent)][1], all_indicesB[int(ievent)][2], predictionsB[int(ievent)][0]))
        if ievent == firstWrong: bkgOut = line
        else: bkgOut = np.concatenate((bkgOut, line))
        #if firstWrong == len(all_indicesB)-1: bkgOut = np.array((-1, -1, -1, -1))
    
    #for i in range(len(e_false)):
    #    index = int(e_false[i])
    #    print(index)
    #    line = np.array((index, 1, predictionsE[index, 0]))
    #    line = np.reshape(line, (1, 3))
    #    if i ==0: eOut = line
    #    else: eOut = np.concatenate((eOut, line))

    #for i in range(len(bkg_false)):
    #    index = int(bkg_false[i])
    #    line = np.array((index, 0, predictionsB[index, 0]))
    #    line= np.reshape(line, (1,3))
    #    if i ==0: bkgOut = line
    #    else: bkgOut = np.concatenate((bkgOut, line))
    
    #if len(e_false) != 0 and len(bkg_false) != 0: allOut = np.concatenate((eOut, bkgOut))
    #elif len(e_false) == 0: allOut = bkgOut
    #elif len(bkg_false) ==0: allOut = eOut
    #else: allOut = []
    
    #allOut = np.concatenate((eOut, bkgOut))
    np.save(batchDir + "falseEventsE.npy", eOut)
    np.save(batchDir + "falseEventsB.npy", bkgOut)

    utils.metrics(true, predictions, plotDir, threshold=0.5)

    print()
    print(utils.bcolors.HEADER+"Validated on "+str(validatedE)+" electron events and "+str(validatedBkg)+" background events"+utils.bcolors.ENDC)
    print(utils.bcolors.GREEN+"Saved metrics to "+plotDir+utils.bcolors.ENDC)
    print()


if __name__ == "__main__":

	dataDir = "/data/disappearingTracks/cleaned/N0p1/"
	batchDir = "/home/mcarrigan/disTracksML/outputFiles/cnn/"
	plotDir = "/home/llavezzo/plots/cnn/"
	valFile = "valBatches"
	tag = "0p25_tanh_"

	input_shape = (40,40,3)

	model = cnn.build_model(input_shape = input_shape, 
						layers = 5, filters = 64, opt='adam')

	validate(model, valFile, batchDir, dataDir, tag, plotDir)
