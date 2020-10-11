import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import random
import sys
import pickle
import datetime
import getopt

def load_data(files, events, class_label, dataDir):
	lastFile = len(files)-1
	images = np.array([])
	for iFile, file in enumerate(files):
		fname = "events_"+str(file)+".npz"
		if(file == -1): continue

		if(iFile == 0 and iFile != lastFile):
			images = np.load(dataDir+fname)[class_label][events[0]:]

		elif(iFile == lastFile and iFile != 0):
			images = np.vstack((images,np.load(dataDir+fname)[class_label][:events[1]+1]))

		elif(iFile == 0 and iFile == lastFile):
			images = np.load(dataDir+fname)[class_label][events[0]:events[1]+1]

		elif(iFile != 0 and iFile != lastFile):
			images = np.vstack((images,np.load(dataDir+fname)[class_label]))

	return images

# generate batches of images from files
class generator(keras.utils.Sequence):
  
	def __init__(self, batchesE, batchesBkg, indicesE, indicesBkg, 
				batch_size, dataDir, val_mode=False, shuffle=True, eventInfo=False):
		self.batchesE = batchesE
		self.batchesBkg = batchesBkg
		self.indicesE = indicesE
		self.indicesBkg = indicesBkg
		self.batch_size = batch_size
		self.dataDir = dataDir
		self.shuffle = shuffle
		self.val_mode = val_mode
		self.y_batches = np.array([])
		self.used_idx = []
		self.indices_batches = np.array([])
		self.eventInfo = eventInfo

	def __len__(self):
		return len(self.batchesE)

	def __getitem__(self, idx) :

		filenamesE = self.batchesE[idx]
		filenamesBkg = self.batchesBkg[idx]
		indexE = self.indicesE[idx]
		indexBkg = self.indicesBkg[idx]

		e_images = load_data(filenamesE,indexE,'signal',self.dataDir)
		bkg_images = load_data(filenamesBkg,indexBkg,'bkg',self.dataDir)
		if(self.eventInfo):
			e_info = load_data(filenamesE,indexE,'s_infos',self.dataDir)
			bkg_info = load_data(filenamesBkg,indexBkg,'bkg_infos',self.dataDir)
		
		numE = e_images.shape[0]
		numBkg = bkg_images.shape[0]

		# shuffle and select appropriate amount of electrons, bkg
		if(numBkg != 0):
			indices = list(range(bkg_images.shape[0]))
			random.shuffle(indices)
			bkg_indices = bkg_images[indices,:4]
			bkg_images = bkg_images[indices,4:]
			if(self.eventInfo): bkg_info = bkg_info[indices,[6,10,11,12,13]]

		if(numE != 0):
			indices = list(range(e_images.shape[0]))
			random.shuffle(indices)
			e_indices = e_images[indices,:4]
			e_images = e_images[indices,4:]
			if(self.eventInfo): e_info = e_info[indices,[6,10,11,12,13]]

		# concatenate images and suffle them, create labels
		if(numE != 0 and numBkg != 0): 
			batch_x = np.vstack((e_images,bkg_images))
			batch_indices = np.vstack((e_indices,bkg_indices))
			batch_y = np.concatenate((np.ones(numE),np.zeros(numBkg)))
			if(self.eventInfo): batch_info = np.vstack((e_info,bkg_info))
		elif(numE == 0): 
			batch_x = bkg_images
			batch_indices = bkg_indices
			batch_y = np.zeros(numBkg)
			if(self.eventInfo): batch_info = bkg_info
		elif(numBkg == 0): 
			batch_x = e_images
			batch_indices = e_indices
			batch_y = np.ones(numE)
			if(self.eventInfo): batch_info = e_info
		
		indices = list(range(batch_x.shape[0]))
		random.shuffle(indices)

		batch_x = batch_x[indices[:self.batch_size],:]
		nEvents = int(batch_x.shape[1]*1.0/4)
		batch_x = np.reshape(batch_x,(self.batch_size,nEvents,4))

		batch_indices = batch_indices[indices[:self.batch_size],:]

		batch_y = batch_y[indices[:self.batch_size]]
		batch_y = keras.utils.to_categorical(batch_y, num_classes=2)

		if(self.eventInfo): batch_info = batch_info[indices[:self.batch_size]]
		
		
		if(self.val_mode):
			if(idx not in self.used_idx):
				if(len(self.used_idx)==0): 
					self.y_batches = batch_y
					self.indices_batches = batch_indices
				else: 
					self.y_batches = np.concatenate((self.y_batches, batch_y))
					self.indices_batches = np.concatenate((self.indices_batches, batch_indices))
				self.used_idx.append(idx)
			
				if(self.eventInfo): return [batch_x, batch_info]
				else: return batch_x

		elif(self.eventInfo):
			return [batch_x, batch_info], batch_y
		else:
			return batch_x, batch_y
	
	def reset(self):
		self.y_batches = np.array([])
		self.used_idx = []

	def get_y_batches(self):
		return self.y_batches

	def get_indices_batches(self):
		return self.indices_batches

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