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
class generator(keras.utils.Sequence):
  
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

        e_images = load_data(filenamesE, indexE, 'e_0p25_', self.dataDir)
        bkg_images = load_data(filenamesBkg, indexBkg, 'bkg_0p25_', self.dataDir)
        
        numE = e_images.shape[0]
        numBkg = self.batch_size-numE
        bkg_images = bkg_images[:numBkg]

        # shuffle and select appropriate amount of electrons, bkg
        indices = list(range(bkg_images.shape[0]))
        random.shuffle(indices)
        bkg_indices = bkg_images[indices,:2]
        bkg_images = bkg_images[indices,2:]

        eOut = np.array([])
        bkgOut = np.array([])

        if(numE != 0):
            indices = list(range(e_images.shape[0]))
            random.shuffle(indices)
            e_indices = e_images[indices,:2]
            e_images = e_images[indices,2:]
            for ievt, event in enumerate(e_indices):
                if ievt == 0: eOut = np.array((e_indices[ievt,0], e_indices[ievt,1], 1))
                else: eOut = np.concatenate((eOut, np.array((e_indices[ievt,0], e_indices[ievt,1], 1))))

        for ievt, event in enumerate(bkg_indices):
            if ievt == 0: bkgOut = np.array((bkg_indices[ievt,0], bkg_indices[ievt,1], 0))
            else: bkgOut = np.concatenate((bkgOut, np.array((bkg_indices[ievt,0], bkg_indices[ievt,1], 0))))
        
        eOut = np.reshape(eOut, (-1,3))
        bkgOut = np.reshape(bkgOut, (-1,3))

        # concatenate images and suffle them, create labels
        if(numE != 0): batch_x = np.vstack((e_images,bkg_images))
        else: batch_x = bkg_images
        batch_y = np.concatenate((np.ones(numE),np.zeros(numBkg)))
        allOut = np.concatenate((eOut, bkgOut))
        indices = list(range(batch_x.shape[0]))
        random.shuffle(indices)

        batch_x = batch_x[indices[:self.batch_size],:]
        batch_x = batch_x[:self.batch_size]
        print(batch_x.shape)
        print(allOut.shape)
        allOut = allOut[:self.batch_size]
        print(allOut.shape)
        batch_x = np.reshape(batch_x,(self.batch_size,40,40,4))
        batch_x = batch_x[:,:,:,[0,2,3]]
        allOut = allOut[indices]

        #if(os.path.exists(self.outputDir+'indexValidate.npy')):
        #    all_indices = np.load(self.outputDir+'indexValidate.npy')
        #    all_indices = np.concatenate((all_indices, allOut))
        #    np.save(self.outputDir+'indexValidate.npy', all_indices)
        #    print("all indices shape: ", all_indices.shape)
        #else: np.save(self.outputDir+'indexValidate.npy', allOut)

        batch_y = batch_y[indices[:self.batch_size]]
        #batch_y = keras.utils.to_categorical(batch_y, num_classes=2)
        
        if(self.save_truth_labels):
            if(idx not in self.used_idx):
                self.y_batches = np.append(self.y_batches, batch_y)
                self.used_idx.append(idx)
                self.indices_used = np.reshape(self.indices_used, (-1, 3))
                print("self indices shape", self.indices_used.shape)
                print("array shape", allOut.shape)
                self.indices_used = np.concatenate((self.indices_used, allOut))

        if(self.return_y_batches):
            return batch_x, batch_y
        else:
            return batch_x
    
    def on_epoch_end(self):
        if(self.shuffle):
            indexes = np.arange(len(self.batchesE))
            np.random.shuffle(indexes)
            self.batchesE = batchesE[indexes]
            self.batchesBkg = batchesBkg[indexes]
            self.indicesE = indicesE[indexes]
            self.indicesBkg = indicesBkg[indexes]

    def reset(self):
        self.y_batches = np.array([])
        self.used_idx = []

    def get_y_batches(self):
        return self.y_batches

    def get_indices_used(self):
        self.indices_used = self.indices_used[1:]
        return self.indices_used


def validate(model, weights, batchDir, dataDir, plotDir):
    print("------------------STARTING VALIDATION--------------------")
    model.load_weights(weights)

    # load the batches used to train and validate
    val_e_file_batches = np.load(batchDir+'e_files_valBatches.npy', allow_pickle=True)
    val_e_event_batches = np.load(batchDir+'e_events_valBatches.npy', allow_pickle=True)
    val_bkg_file_batches = np.load(batchDir+'bkg_files_valBatches.npy', allow_pickle=True)
    val_bkg_event_batches = np.load(batchDir+'bkg_events_valBatches.npy', allow_pickle=True)

    batch_size = 32
    print("Define Generator")
    val_generator = generator(val_e_file_batches, val_bkg_file_batches, val_e_event_batches, val_bkg_event_batches, batch_size, dataDir, batchDir)
    print("reset generator")
    val_generator.reset()
    print("Get predictions")
    predictions = model.predict(val_generator, verbose=1)
    true = val_generator.get_y_batches()
    indices = val_generator.get_indices_used()

    # make sure utils.metrics works
    cm = np.zeros((2,2))
    for p,t in zip(predictions, true):
        pr = int(np.rint(p))
        tr = int(np.rint(t))
        cm[pr][tr]+=1
    print(cm)

    utils.metrics(true, predictions, plotDir, threshold=0.5)

    #indices = np.load(batchDir+'indexValidate.npy')
    print("indices shape", indices.shape)
    #indices = indices[batch_size:]
    eOut = np.array([])
    bkgOut = np.array([])
    #print(indices.shape, indices.shape[0])
    first_e = 0
    first_b = 0
    count_wrong = 0
    for ievt, event in enumerate(indices):
        if indices[ievt, 2] != true[ievt]: 
            #print("file truth does not match ", indices[ievt, 2], true[ievt])
            count_wrong += 1
            print("Wrong Event", count_wrong)
        print("file output", indices[ievt, 2], "truth", true[ievt])
        if indices[ievt, 2] == 0:
            if first_b == 0:
                bkgOut = np.array((indices[ievt, 0], indices[ievt, 1], indices[ievt, 2], predictions[ievt]))
                first_b = 1
            else: bkgOut = np.concatenate((bkgOut, np.array((indices[ievt, 0], indices[ievt, 1], indices[ievt, 2], predictions[ievt]))))
        if indices[ievt, 2] == 1:
            if first_e == 0:
                eOut = np.array((indices[ievt, 0], indices[ievt, 1], indices[ievt, 2], predictions[ievt]))
                first_e = 1
            else: eOut = np.concatenate((eOut, np.array((indices[ievt, 0], indices[ievt, 1], indices[ievt, 2], predictions[ievt]))))
    np.save(batchDir+"falseEventsE.npy", eOut)
    np.save(batchDir+"falseEventsB.npy", bkgOut)



    print()
    print(utils.bcolors.GREEN+"Saved metrics to "+plotDir+utils.bcolors.ENDC)
    print()


if __name__ == "__main__":

    dataDir = "/data/disappearingTracks/electron_selection/"
    batchDir = "undersample0p5/outputFiles/"
    weights = "undersample0p5/weights/weights.30.h5"
    plotDir = "undersample0p5/"

    model = cnn.build_model(input_shape = (40,40,3), 
                        layers = 3, filters = 64, opt='adam')

    validate(model, weights, batchDir, dataDir, plotDir)
