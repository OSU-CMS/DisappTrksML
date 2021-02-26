import numpy as np
import glob, os, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deepSetsMuons import *

# initialize the model with the weights
fileDir = 'train_calos/'
model_file = 'model.16.h5'
arch = DeepSetsArchitecture()
arch.load_model(fileDir+model_file)

cm = np.zeros((2,2))

sets, infos = None, None

count = 0
# evaluate the model
bkg_preds = None
bkgDir = "/store/user/llavezzo/disappearingTracks/SingleMu_2017F_wIso_withCalos/"
inputFiles = glob.glob(bkgDir+'images_*.root.npz')
for i,fname in enumerate(inputFiles):
	print(i)
	if i == 500: break

	data = np.load(fname, allow_pickle=True)
	count += data['sets'].shape[0]

	skip, preds = arch.evaluate_npy(fname, True, False,'sets')
	if not skip:
		cm[0,1] += np.count_nonzero(preds[:,1] > 0.5)
		cm[0,0] += np.count_nonzero(preds[:,1] <= 0.5)

	print cm

print count