import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DisappTrksML.DeepSets.architecture import *

# parameters
dataDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_converted/"
modelFile = 'kfold0/model.18.h5'
outDir = "kfold0/"
fileRange = [0,100]				# set to -1 if not interested

# initialize architecture and load the weights/model
arch = DeepSetsArchitecture(maxHits=100)
arch.load_model(modelFile)

# iterate over desired files and predict
cm = np.zeros((2,2))
inputFiles = glob.glob(dataDir+'images_*.root.npz')
for iFile in inputFiles:
	print(i)

	# evaluate file
	signal_preds, skip = arch.evaluate_npy(iFile, track_info=False, obj='signal')
	bkg_preds, skip = arch.evaluate_npy(iFile, track_info=False, obj='background')
	if skip: continue

	# fill confusion matrix
	cm[1,1] += np.count_nonzero(signal_preds[:,1] > 0.5)
	cm[1,0]+= np.count_nonzero(signal_preds[:,1] <= 0.5)
	cm[0,1] += np.count_nonzero(bkg_preds[:,1] > 0.5)
	cm[0,0]+= np.count_nonzero(bkg_preds[:,1] <= 0.5)

# calculate metrics
precision, recall, f1 = arch.calc_binary_metrics(cm)

# count number of total predictions
nPreds = np.sum(cm)

# save to file
metrics = [precision, recall, f1, nPreds]
np.savetxt(outDir + "/evaluation_metrics.csv", metrics, delimiter=",")