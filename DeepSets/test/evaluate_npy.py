import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DisappTrksML.DeepSets.architecture import *

# initialize the model with the weights
fileDir = 'kfold2/'
model_file = 'model.19.h5'
arch = DeepSetsArchitecture()
arch.load_model(fileDir+model_file)

# predicting
cm = np.zeros((2,2))
bkgDir = "/store/user/llavezzo/disappearingTracks/images_DYJetsToLL_v5_converted/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')
inputIndices = np.array([f.split(bkgDir+'images_')[-1][:-9] for f in inputFiles])
for i in range(600,700):
	print(i)
	data = np.load(bkgDir+'images_'+str(inputIndices[i])+'.root.npz',allow_pickle=True)

	signal = data['signal']
	bkg = data['background']

	signal_preds = arch.model.predict(signal)
	bkg_preds = arch.model.predict(bkg)

	cm[1,1] += np.count_nonzero(signal_preds[:,1] > 0.5)
	cm[1,0]+= np.count_nonzero(signal_preds[:,1] <= 0.5)
	cm[0,1] += np.count_nonzero(bkg_preds[:,1] > 0.5)
	cm[0,0]+= np.count_nonzero(bkg_preds[:,1] <= 0.5)

print cm