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

def validate(model, weights, batchDir, dataDir, plotDir):

	model.load_weights(weights)

	# load the batches used to train and validate
	val_e_file_batches = np.load(batchDir+'e_files_valBatches.npy', allow_pickle=True)
	val_e_event_batches = np.load(batchDir+'e_events_valBatches.npy', allow_pickle=True)
	val_bkg_file_batches = np.load(batchDir+'bkg_files_valBatches.npy', allow_pickle=True)
	val_bkg_event_batches = np.load(batchDir+'bkg_events_valBatches.npy', allow_pickle=True)

	batch_size = 512

	val_generator = cnn.generator(val_e_file_batches, val_bkg_file_batches, val_e_event_batches, val_bkg_event_batches, batch_size, dataDir, False)
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
	weights = "undersample0p5/weights/weights.37.h5"
	plotDir = "undersample0p5/"

	model = cnn.build_model(input_shape = (40,40,3), 
						layers = 3, filters = 64, opt='adam')

	validate(model, weights, batchDir, dataDir, plotDir)