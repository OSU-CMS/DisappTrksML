import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from tensorflow.keras.applications import VGG19
import json
import random
import sys
import pickle
import datetime
import getopt

# generate batches of images from files
class generator(keras.utils.Sequence):
  
	def __init__(self, batchesE, batchesBkg, indicesE, indicesBkg, 
				batch_size, dataDir, shuffle=True):
		self.batchesE = batchesE
		self.batchesBkg = batchesBkg
		self.indicesE = indicesE
		self.indicesBkg = indicesBkg
		self.batch_size = batch_size
		self.dataDir = dataDir
		self.shuffle = shuffle

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
		
		return batch_x, batch_y
	
	# def on_epoch_end(self):
	# 	if(self.shuffle):
	# 		indices = np.arange(len(self.batchesE)).astype(int)
	# 		np.random.shuffle(indices)
	# 		self.batchesE = [self.batchesE[i] for i in indices]
	# 		self.indicesE = [self.indicesE[i] for i in indices]
	# 		indices = np.arange(len(self.batchesBkg)).astype(int)
	# 		np.random.shuffle(indices)
	# 		self.batchesBkg = [self.batchesBkg[i] for i in indices]
	# 		self.indicesBkg = [self.indicesBkg[i] for i in indices]