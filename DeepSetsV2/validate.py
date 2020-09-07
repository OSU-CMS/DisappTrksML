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
from generator import generator 


def run_validation(model, weights, batchDir, dataDir, plotDir, batch_size):
    print("------------------STARTING VALIDATION--------------------")
    model.load_weights(weights)

    # load the batches used to train and validate
    val_e_file_batches = np.load(batchDir+'e_files_valBatches.npy', allow_pickle=True)
    val_e_event_batches = np.load(batchDir+'e_events_valBatches.npy', allow_pickle=True)
    val_bkg_file_batches = np.load(batchDir+'bkg_files_valBatches.npy', allow_pickle=True)
    val_bkg_event_batches = np.load(batchDir+'bkg_events_valBatches.npy', allow_pickle=True)

    print("Define Generator")
    val_generator = generator(val_e_file_batches, val_bkg_file_batches, val_e_event_batches, val_bkg_event_batches, 
        batch_size, dataDir, True, False)
    print("reset generator")
    val_generator.reset()
    print("Get predictions")
    predictions = model.predict(val_generator, verbose=2)
    true = val_generator.get_y_batches()

    cm = np.zeros((2,2)) 
    for t,pred in zip(true,predictions):
        if(pred[1]>0.5):
            if(t[1]>0.5): cm[1][1]+=1;
            else: cm[1][0]+=1;
        else:
            if(t[1]>0.5): cm[0][1]+=1;
            else: cm[0][0]+=1;
    print(cm)

    utils.metrics(true[:,1], predictions[:,1], plotDir, threshold=0.5)

    # eOut = np.array([])
    # bkgOut = np.array([])
    # first_e = 0
    # first_b = 0
    # count_wrong = 0
    # for ievt, event in enumerate(indices):
    #     if indices[ievt, 2] != true[ievt]: 
    #         count_wrong += 1
    #     if indices[ievt, 2] == 0:
    #         if first_b == 0:
    #             bkgOut = np.array((indices[ievt, 0], indices[ievt, 1], indices[ievt, 2], predictions[ievt]))
    #             first_b = 1
    #         else: bkgOut = np.concatenate((bkgOut, np.array((indices[ievt, 0], indices[ievt, 1], indices[ievt, 2], predictions[ievt]))))
    #     if indices[ievt, 2] == 1:
    #         if first_e == 0:
    #             eOut = np.array((indices[ievt, 0], indices[ievt, 1], indices[ievt, 2], predictions[ievt]))
    #             first_e = 1
    #         else: eOut = np.concatenate((eOut, np.array((indices[ievt, 0], indices[ievt, 1], indices[ievt, 2], predictions[ievt]))))
    # np.save(batchDir+"falseEventsE.npy", eOut)
    # np.save(batchDir+"falseEventsB.npy", bkgOut)


    print()
    print(utils.bcolors.GREEN+"Saved metrics to "+plotDir+utils.bcolors.ENDC)
    print()


if __name__ == "__main__":

    import train as deepSet

    dataDir = "/store/user/llavezzo/disappearingTracks/converted_deepSets_failAllRecos/"
    batchDir = "deepSets_2/outputFiles/"
    weights = "deepSets_2/weights/lastEpoch.h5"
    plotDir = "deepSets_2/"

    model = deepSet.buildModel()

    model.compile(optimizer=optimizers.Adam(), 
                  loss='categorical_crossentropy', 
                  metrics=metrics)

    run_validation(model, weights, batchDir, dataDir, plotDir)
