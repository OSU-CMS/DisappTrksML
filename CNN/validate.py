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
				batch_size, dataDir, return_y_batches=True, save_truth_labels=True):
		self.batchesE = batchesE
		self.batchesBkg = batchesBkg
		self.indicesE = indicesE
		self.indicesBkg = indicesBkg
		self.batch_size = batch_size
		self.dataDir = dataDir
		self.y_batches = np.array([])				# truth labels
		self.used_idx = []							# keeps count of which y batches to save
		self.return_y_batches = return_y_batches	# set to False when validating
		self.save_truth_labels = save_truth_labels	# call get_y_batches() to obtain them, reset() to erase them

	def __len__(self):
		return len(self.batchesE)

	def __getitem__(self, idx) :

		filenamesE = self.batchesE[idx]
		filenamesBkg = self.batchesBkg[idx]
		indexE = self.indicesE[idx]
		indexBkg = self.indicesBkg[idx]

		e_images = load_data(filenamesE, indexE, 'e_0p25_', dataDir)
		bkg_images = load_data(filenamesBkg, indexBkg, 'bkg_0p25_', dataDir)
	    
		numE = e_images.shape[0]
		numBkg = self.batch_size-numE
		bkg_images = bkg_images[:numBkg]

		# shuffle and select appropriate amount of electrons, bkg
		indices = list(range(bkg_images.shape[0]))
		random.shuffle(indices)
		bkg_images = bkg_images[indices,1:]

		if(numE != 0):
			indices = list(range(e_images.shape[0]))
			random.shuffle(indices)
			e_images = e_images[indices,1:]

		# concatenate images and suffle them, create labels
		if(numE != 0): batch_x = np.vstack((e_images,bkg_images))
		else: batch_x = bkg_images
		batch_y = np.concatenate((np.ones(numE),np.zeros(numBkg)))

		indices = list(range(batch_x.shape[0]))
		random.shuffle(indices)

		batch_x = batch_x[indices[:self.batch_size],:]
		batch_x = np.reshape(batch_x,(self.batch_size,40,40,4))
		batch_x = batch_x[:,:,:,[0,2,3]]

		batch_y = batch_y[indices[:self.batch_size]]
		#batch_y = keras.utils.to_categorical(batch_y, num_classes=2)
		
		if(self.save_truth_labels):
			if(idx not in self.used_idx):
					self.y_batches = np.append(self.y_batches, batch_y)
					self.used_idx.append(idx)

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


def validate(model, weights, batchDir, dataDir, plotDir):

	model.load_weights(weights)

	# load the batches used to train and validate
	val_e_file_batches = np.load(batchDir+'e_files_valBatches.npy', allow_pickle=True)
	val_e_event_batches = np.load(batchDir+'e_events_valBatches.npy', allow_pickle=True)
	val_bkg_file_batches = np.load(batchDir+'bkg_files_valBatches.npy', allow_pickle=True)
	val_bkg_event_batches = np.load(batchDir+'bkg_events_valBatches.npy', allow_pickle=True)

	batch_size = 512

	val_generator = generator(val_e_file_batches[:2], val_bkg_file_batches[:2], val_e_event_batches[:2], val_bkg_event_batches[:2], batch_size, dataDir, False)
	val_generator.reset()
	predictions = model.predict(val_generator, verbose=1)
	true = val_generator.get_y_batches()

	# make sure utils.metrics works
	cm = np.zeros((2,2))
	for p,t in zip(predictions, true):
		pr = int(np.rint(p))
		tr = int(np.rint(t))
		cm[pr][tr]+=1
	print(cm)

	utils.metrics(true, predictions, plotDir, threshold=0.5)

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