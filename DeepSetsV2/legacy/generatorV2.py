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

def load_data(batch, dataDir):
	
	images, labels = [], []
	for event in batch:
		if(event[0]==1):
			images.append(np.load(dataDir+"images_0p5_"+str(event[1])+".npz")['e'][event[2]])
			labels.append(1)
		elif(event[0]==0):
			images.append(np.load(dataDir+"images_0p5_"+str(event[1])+".npz")['bkg'][event[2]])
			labels.append(0)

	return np.array(images), np.array(labels)

# generate batches of images from files
class generator(keras.utils.Sequence):
  
	def __init__(self, batches, batch_size, dataDir, val_mode=False, shuffle=True):
		self.batches = batches
		self.batch_size = batch_size
		self.dataDir = dataDir
		self.shuffle = shuffle
		self.val_mode = val_mode
		self.y_batches = np.array([])
		self.used_idx = []
		self.indices_batches = np.array([])

	def __len__(self):
		return len(self.batches)

	def __getitem__(self, idx) :

		batch = self.batches[idx]

		batch_x, batch_y = load_data(batch, self.dataDir)
		
		indices = list(range(batch_x.shape[0]))
		random.shuffle(indices)

		batch_x = batch_x[indices[:self.batch_size],2:]
		nEvents = int(batch_x.shape[1]*1.0/4)
		batch_x = np.reshape(batch_x,(self.batch_size,nEvents,4))

		batch_indices = batch_x[indices[:self.batch_size],:2]

		batch_y = batch_y[indices[:self.batch_size]]
		batch_y = keras.utils.to_categorical(batch_y, num_classes=2)
		
		
		if(self.val_mode):
			if(idx not in self.used_idx):
				if(len(self.used_idx)==0): 
					self.y_batches = batch_y
					self.indices_batches = batch_indices
				else: 
					self.y_batches = np.concatenate((self.y_batches, batch_y))
					self.indices_batches = np.concatenate((self.indices_batches, batch_indices))
				self.used_idx.append(idx)
			
				return batch_x

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