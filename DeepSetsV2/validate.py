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

def load_data(files, events, label, dataDir):
    lastFile = len(files)-1
    files.sort()
    for iFile, file in enumerate(files):
        if(file == -1): 
            images = np.array([])
            continue
        if(iFile == 0 and iFile != lastFile):
            images = np.load(dataDir+label+str(file)+'.npy')[events[0]:]

        elif(iFile == lastFile and iFile != 0):
            images = np.vstack((images,np.load(dataDir+label+str(file)+'.npy')[:events[1]+1]))

        elif(iFile == 0 and iFile == lastFile):
            images = np.load(dataDir+label+str(file)+'.npy')[events[0]:events[1]+1]

        elif(iFile != 0 and iFile != lastFile):
            images = np.vstack((images,np.load(dataDir+label+str(file)+'.npy')))
    return images

# generate batches of images from files
class validation_generator(keras.utils.Sequence):
  
    def __init__(self, batchesE, batchesBkg, indicesE, indicesBkg, 
                batch_size, dataDir, outputDir, return_y_batches=True, save_truth_labels=True):
        self.batchesE = batchesE
        self.batchesBkg = batchesBkg
        self.indicesE = indicesE
        self.indicesBkg = indicesBkg
        self.batch_size = batch_size
        self.dataDir = dataDir
        self.outputDir = outputDir
        self.y_batches = np.array([])
        self.indices_used = np.array([-1, -1, -1])               
        self.used_idx = []                            # keeps count of which y batches to save
        self.return_y_batches = return_y_batches    # set to False when validating
        self.save_truth_labels = save_truth_labels    # call get_y_batches() to obtain them, reset() to erase them

    def __len__(self):
        return len(self.batchesE)

    def __getitem__(self, idx) :

        filenamesE = self.batchesE[idx]
        filenamesBkg = self.batchesBkg[idx]
        indexE = self.indicesE[idx]
        indexBkg = self.indicesBkg[idx]

        lastFile = len(filenamesE)-1
        filenamesE.sort()
        for iFile, file in enumerate(filenamesE):

            fname = "images_0p5_"+str(file)+".npz"
            if(file == -1): 
                e_images = np.array([])
                continue

            if(iFile == 0 and iFile != lastFile):
                e_images = np.load(self.dataDir+fname)['e'][indexE[0]:]

            elif(iFile == lastFile and iFile != 0):
                e_images = np.vstack((e_images,np.load(self.dataDir+fname)['e'][:indexE[1]+1]))

            elif(iFile == 0 and iFile == lastFile):
                e_images = np.load(self.dataDir+fname)['e'][indexE[0]:indexE[1]+1]

            elif(iFile != 0 and iFile != lastFile):
                e_images = np.vstack((e_images,np.load(self.dataDir+fname)['e']))
        
        lastFile = len(filenamesBkg)-1
        filenamesBkg.sort()
        for iFile, file in enumerate(filenamesBkg):

            fname = "images_0p5_"+str(file)+".npz"
            if(iFile == 0 and iFile != lastFile):
                bkg_images = np.load(self.dataDir+fname)['bkg'][indexBkg[0]:,:]

            elif(iFile == lastFile and iFile != 0):
                bkg_images = np.vstack((bkg_images,np.load(self.dataDir+fname)['bkg'][:indexBkg[1]+1]))

            elif(iFile == 0 and iFile == lastFile):
                bkg_images = np.load(self.dataDir+fname)['bkg'][indexBkg[0]:indexBkg[1]+1]

            elif(iFile != 0 and iFile != lastFile):
                bkg_images = np.vstack((bkg_images,np.load(self.dataDir+fname)['bkg']))
        
        numE = e_images.shape[0]
        numBkg = self.batch_size-numE
        bkg_images = bkg_images[:numBkg]

        # shuffle and select appropriate amount of electrons, bkg
        indices = list(range(bkg_images.shape[0]))
        random.shuffle(indices)
        bkg_images = bkg_images[indices,2:]

        if(numE != 0):
            indices = list(range(e_images.shape[0]))
            random.shuffle(indices)
            e_images = e_images[indices,2:]

        # concatenate images and suffle them, create labels
        if(numE != 0): batch_x = np.vstack((e_images,bkg_images))
        else: batch_x = bkg_images
        batch_y = np.concatenate((np.ones(numE),np.zeros(numBkg)))

        indices = list(range(batch_x.shape[0]))
        random.shuffle(indices)

        batch_x = batch_x[indices[:self.batch_size],:]
        nEvents = int(batch_x.shape[1]*1.0/4)
        batch_x = np.reshape(batch_x,(self.batch_size,nEvents,4))

        batch_y = batch_y[indices[:self.batch_size]]
        batch_y = keras.utils.to_categorical(batch_y, num_classes=2)

        if(idx not in self.used_idx):
            if(len(self.used_idx)==0):  self.y_batches = batch_y
            else: self.y_batches = np.concatenate((self.y_batches, batch_y))
            self.used_idx.append(idx)
        
        return batch_x

    def reset(self):
        self.y_batches = np.array([])
        self.used_idx = []

    def get_y_batches(self):
        return self.y_batches


def run_validation(model, weights, batchDir, dataDir, plotDir, batch_size):
    print("------------------STARTING VALIDATION--------------------")
    model.load_weights(weights)

    # load the batches used to train and validate
    val_e_file_batches = np.load(batchDir+'e_files_valBatches.npy', allow_pickle=True)
    val_e_event_batches = np.load(batchDir+'e_events_valBatches.npy', allow_pickle=True)
    val_bkg_file_batches = np.load(batchDir+'bkg_files_valBatches.npy', allow_pickle=True)
    val_bkg_event_batches = np.load(batchDir+'bkg_events_valBatches.npy', allow_pickle=True)

    print("Define Generator")
    val_generator = validation_generator(val_e_file_batches, val_bkg_file_batches, val_e_event_batches, val_bkg_event_batches, batch_size, dataDir, batchDir)
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
